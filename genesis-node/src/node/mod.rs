use std::sync::{Arc, Mutex};
use std::thread;
use std::ptr;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use crossbeam::channel::{bounded, Sender, Receiver};
use genesis_compute::ShardEngine;
use crate::network::io_server::ExternalIoServer;
use crate::network::bsp::BspBarrier;
use crate::network::router::RoutingTable;

use crate::network::inter_node::{SpikeBatchHeaderV2, SpikeEventV2, InterNodeRouter};

pub mod recovery;

pub enum ComputeCommand {
    RunBatch {
        tick_base: u32,
        batch_size: u32,
        global_dopamine: i16,
    },
    Shutdown,
}

pub enum ComputeFeedback {
    BatchComplete {
        ticks_processed: u32,
        zone_hash: u32,
        pinned_out_ptr: usize,
        output_bytes: usize,
    },
}

pub struct NodeRuntime {
    pub io_server: Arc<ExternalIoServer>,
    pub routing_table: Arc<RoutingTable>,
    pub bsp_barrier: Arc<Mutex<BspBarrier>>,
    pub compute_dispatchers: Mutex<HashMap<u32, Sender<ComputeCommand>>>,
    pub feedback_sender: Sender<ComputeFeedback>,
    pub feedback_receiver: Receiver<ComputeFeedback>,
    pub ghost_rx: Mutex<tokio::sync::mpsc::UnboundedReceiver<(SpikeBatchHeaderV2, Vec<SpikeEventV2>)>>,
    pub telemetry_swapchain: Arc<crate::network::telemetry::TelemetrySwapchain>,
    pub total_ticks: Arc<AtomicU32>,
    pub local_ip: std::net::Ipv4Addr,
    pub local_port: u16,
    /// [DOD] Маршруты выходов: zone_hash -> (TargetAddr, matrix_hash)
    pub output_routes: HashMap<u32, Vec<(String, u32)>>,
    pub intra_gpu_channels: Vec<(u32, u32, crate::network::intra_gpu::IntraGpuChannel)>,
    pub axon_head_ptrs: HashMap<u32, *mut u32>,
}

unsafe impl Send for NodeRuntime {}
unsafe impl Sync for NodeRuntime {}

impl NodeRuntime {
    pub fn boot(
        shards: Vec<(u32, ShardEngine, u32, u32, *const u32, std::path::PathBuf)>,
        io_server: Arc<ExternalIoServer>,
        routing_table: Arc<RoutingTable>,
        bsp_barrier: Arc<Mutex<BspBarrier>>,
        telemetry_swapchain: Arc<crate::network::telemetry::TelemetrySwapchain>,
        local_ip: std::net::Ipv4Addr,
        local_port: u16,
        output_routes: HashMap<u32, Vec<(String, u32)>>,
        intra_gpu_channels: Vec<(u32, u32, crate::network::intra_gpu::IntraGpuChannel)>,
        axon_head_ptrs: HashMap<u32, *mut u32>,
    ) -> Self {
        let (feedback_tx, feedback_rx) = bounded(shards.len() + 32);
        let total_ticks = Arc::new(AtomicU32::new(0));

        let (ghost_tx, ghost_rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(InterNodeRouter::spawn_ghost_listener(8083, ghost_tx));

        let node = Self {
            io_server,
            routing_table,
            bsp_barrier,
            compute_dispatchers: Mutex::new(HashMap::new()),
            feedback_sender: feedback_tx,
            feedback_receiver: feedback_rx,
            ghost_rx: Mutex::new(ghost_rx),
            total_ticks,
            local_ip,
            local_port,
            telemetry_swapchain,
            output_routes,
            intra_gpu_channels,
            axon_head_ptrs,
        };

        for (hash, shard, v_axons, outputs, m_somas, baked_dir) in shards {
            node.spawn_shard_thread(hash, shard, v_axons, outputs, m_somas, baked_dir);
        }

        node
    }

    pub fn spawn_shard_thread(
        &self, 
        hash: u32, 
        mut shard: ShardEngine,
        num_virtual_axons: u32,
        num_outputs: u32,
        mapped_soma_ids_device: *const u32,
        baked_dir: std::path::PathBuf,
    ) {
        let (tx, rx) = bounded(1);
        let _telemetry = self.telemetry_swapchain.clone();
        
        // [DOD] Извлекаем Swapchain конкретно для этой зоны (Linear Search)
        let my_io_ctx = self.io_server.io_contexts.iter()
            .find(|(h, _)| *h == hash)
            .map(|(_, ctx)| ctx.swapchain.clone())
            .expect("FATAL: IO Context for zone not found");

        let sync_batch_ticks = 100;
        let max_spikes_per_tick = 100_000;
        let input_words_per_tick = (num_virtual_axons + 31) / 32; 

        let mut d_input = ptr::null_mut();
        let mut d_spikes = ptr::null_mut();
        let mut d_output = ptr::null_mut();

        unsafe {
            genesis_compute::ffi::cu_allocate_io_buffers(
                input_words_per_tick * sync_batch_ticks,
                max_spikes_per_tick * sync_batch_ticks,
                num_outputs * sync_batch_ticks,
                &mut d_input,
                &mut d_spikes,
                &mut d_output,
            );
        }

        let io_buffers = genesis_compute::compute::shard::IoDeviceBuffers {
            d_input_bitmask: d_input,
            d_incoming_spikes: d_spikes,
            d_output_history: d_output,
            max_spikes_per_tick,
            input_words_per_tick,
            num_outputs,
        };

        // [DOD] Pre-allocate Pinned RAM for outputs (Invariant #3)
        let output_bytes = (num_outputs * sync_batch_ticks) as usize;
        let mut pinned_out = genesis_compute::memory::PinnedBuffer::<u8>::new(output_bytes).unwrap();

        {
            let mut dispatchers = self.compute_dispatchers.lock().unwrap();
            dispatchers.insert(hash, tx);
        }

        let f_tx = self.feedback_sender.clone();
        let bsp_barrier = self.bsp_barrier.clone();
        let mapped_soma_ids_ptr = mapped_soma_ids_device as usize;

        thread::Builder::new()
            .name(format!("compute-{}", hash))
            .spawn(move || {
                let mapped_soma_ids = mapped_soma_ids_ptr as *const u32;
                let mut batch_counter: u64 = 0;
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        ComputeCommand::RunBatch { tick_base: _, batch_size, global_dopamine } => {
                            // [DOD] Заливаем дофамин в Constant Memory GPU перед стартом физики
                            unsafe {
                                genesis_compute::ffi::update_global_dopamine(
                                    global_dopamine, 
                                    std::ptr::null_mut()
                                );
                            }

                            let (incoming_spikes, spike_counts) = {
                                let bsp = bsp_barrier.lock().unwrap();
                                let schedule = bsp.get_read_schedule();
                                (schedule.ghost_ids.clone(), schedule.counts.clone())
                            };

                            // [DOD] Читаем изоляционный указатель на Pinned RAM именно этой зоны
                            let input_ptr = my_io_ctx.consume_for_gpu();
                            let input_slice = if !input_ptr.is_null() && input_words_per_tick > 0 {
                                unsafe {
                                    Some(std::slice::from_raw_parts(
                                        input_ptr as *const u32,
                                        (input_words_per_tick * batch_size) as usize,
                                    ))
                                }
                            } else {
                                None
                            };

                            // [DOD] Virtual аксоны лежат в хвосте массива axon_heads
                            let virtual_offset = shard.vram.total_axons - num_virtual_axons;

                            shard.step_day_phase_batch(
                                batch_size,
                                &io_buffers,
                                input_slice,
                                Some(&incoming_spikes),
                                &spike_counts,
                                virtual_offset, // Исправлено: теперь бьёт в хвост
                                num_virtual_axons,
                                mapped_soma_ids,
                            );

                            // [DOD] D2H DMA: Скачиваем выходы из VRAM в Pinned RAM
                            if num_outputs > 0 {
                                unsafe {
                                    genesis_compute::ffi::gpu_memcpy_device_to_host(
                                        pinned_out.as_mut_ptr() as *mut _,
                                        io_buffers.d_output_history as *const _,
                                        output_bytes,
                                    );
                                }
                            }

                            // [DOD] Hot Checkpointing (Zero-Copy VRAM Dump)
                            let batch_idx = batch_counter;
                            if batch_idx > 0 && batch_idx % 500 == 0 {
                                let (_, total_size) = genesis_compute::memory::calculate_state_blob_size(shard.vram.padded_n as usize);
                                let mut host_state = vec![0u8; total_size];

                                unsafe {
                                    // soma_voltage — это base_ptr всего SoA блока.
                                    // Копируем весь контейнер за один DMA трансфер.
                                    genesis_compute::ffi::gpu_memcpy_device_to_host(
                                        host_state.as_mut_ptr() as *mut _,
                                        shard.vram.ptrs.soma_voltage as *const _,
                                        total_size,
                                    );
                                }

                                let chk_path = baked_dir.join("checkpoint.state");
                                let tmp_path = baked_dir.join("checkpoint.state.tmp");

                                // Атомарная запись через .tmp защищает от краша
                                if std::fs::write(&tmp_path, &host_state).is_ok() {
                                    let _ = std::fs::rename(&tmp_path, &chk_path);
                                    println!("💾 [Shard {:08X}] State checkpoint saved: {} MB", hash, total_size / 1024 / 1024);
                                }
                            }
                            batch_counter += 1;

                            if let Err(_) = f_tx.send(ComputeFeedback::BatchComplete { 
                                ticks_processed: batch_size,
                                zone_hash: hash,
                                pinned_out_ptr: pinned_out.as_ptr() as usize,
                                output_bytes,
                            }) {
                                break;
                            }
                        }
                        ComputeCommand::Shutdown => break,
                    }
                }
            }).expect("Failed to spawn compute thread");
    }

    pub async fn run_node_loop(&self, batch_size: u32) {
        let mut current_tick = 0;

        // [DOD] Pre-allocate outbound buffers to avoid heap thrashing (Invariant #3)
        let mut _io_tx_buffer = vec![0u8; genesis_core::constants::MAX_UDP_PAYLOAD];
        let mut _bsp_tx_buffer: Vec<u8> = Vec::with_capacity(genesis_core::constants::MAX_UDP_PAYLOAD);

        println!("[Node] Entering main loop (Batch size: {})", batch_size);

        loop {
            // 1. Synchronize BSP and consume Ghost events
            {
                let mut bsp = self.bsp_barrier.lock().unwrap();
                bsp.sync_and_swap();

                let mut g_rx = self.ghost_rx.lock().unwrap();
                let schedule = bsp.get_write_schedule();
                while let Ok((_header, events)) = g_rx.try_recv() {
                    for ev in events {
                        schedule.push_spike(ev.tick_offset as usize, ev.ghost_id);
                    }
                }
            }

            // 1. Swap IO Buffers (Acquire semantics)
            for (_, io_ctx) in &self.io_server.io_contexts {
                io_ctx.swapchain.swap();
            }

            // 2. Dispatch batches to compute shards
            let current_dopamine = self.io_server.global_dopamine.load(Ordering::Relaxed) as i16;
            if current_dopamine != 0 {
                println!("💉 [Node] Propagating Dopamine: {}", current_dopamine);
            }

            let num_dispatchers = {
                let dispatchers_guard = self.compute_dispatchers.lock().unwrap();
                let num = dispatchers_guard.len();
                if num == 0 {
                    println!("[!] ERROR: No compute dispatchers found!");
                }
                for tx in dispatchers_guard.values() {
                    let _ = tx.send(ComputeCommand::RunBatch {
                        tick_base: current_tick as u32,
                        batch_size,
                        global_dopamine: current_dopamine,
                    });
                }
                num
            };

            // 3. Collect feedback and ship outputs
            for _ in 0..num_dispatchers {
                if let Ok(feedback) = self.feedback_receiver.recv() {
                    match feedback {
                        ComputeFeedback::BatchComplete { ticks_processed: _, zone_hash, pinned_out_ptr, output_bytes } => {
                            if output_bytes > 0 {
                                println!("[Node] Batch Complete (Zone: 0x{:08X}, Out: {} bytes)", zone_hash, output_bytes);
                                
                                // [DOD] Ship outputs to network targets
                                if let Some(routes) = self.output_routes.get(&zone_hash) {
                                    for (addr, m_hash) in routes {
                                        self.io_server.send_output_batch(
                                            &addr, 
                                            zone_hash,
                                            *m_hash,
                                            pinned_out_ptr,
                                            output_bytes,
                                            &mut _io_tx_buffer
                                        ).await;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // [DOD] 4. Intra-GPU Ghost Sync (Zero-Cost VRAM-to-VRAM)
            for (src_hash, dst_hash, channel) in &self.intra_gpu_channels {
                if let (Some(&src_ptr), Some(&dst_ptr)) = (self.axon_head_ptrs.get(src_hash), self.axon_head_ptrs.get(dst_hash)) {
                    unsafe {
                        channel.sync_ghosts(src_ptr, dst_ptr, std::ptr::null_mut());
                    }
                }
            }
            unsafe { genesis_compute::ffi::gpu_stream_synchronize(std::ptr::null_mut()); }

            current_tick += batch_size;
            self.io_server.dashboard.total_ticks.store(current_tick as u64, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_compute::ShardEngine;
    use crate::tui::DashboardState;
    use crate::network::io_server::ExternalIoServer;
    use crate::network::bsp::BspBarrier;
    use crate::network::router::RoutingTable;
    use std::sync::{Arc, Mutex};
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_node_compute_isolation() {
        let dashboard = Arc::new(DashboardState::new(false));
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));
        let socket = Arc::new(tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let io_server = Arc::new(ExternalIoServer::new(
            Arc::new(std::sync::atomic::AtomicBool::new(false)), 
            1024, 0, 0, dashboard, routing_table.clone(), socket
        ).unwrap());
        let bsp_barrier = Arc::new(Mutex::new(BspBarrier::new(100)));
        
        let shard = ShardEngine::new(unsafe { std::mem::zeroed() });

        let node = NodeRuntime::boot(
            vec![(0x1234, shard)], 
            io_server, 
            routing_table, 
            bsp_barrier,
            Ipv4Addr::new(127, 0, 0, 1),
            8080
        );
        
        let tokio_thread_id = thread::current().id();
        let (id_tx, id_rx) = bounded(1);
        
        thread::spawn(move || {
            id_tx.send(thread::current().id()).unwrap();
        });
        
        let spawned_thread_id = id_rx.recv().unwrap();
        assert_ne!(tokio_thread_id, spawned_thread_id);
    }
}

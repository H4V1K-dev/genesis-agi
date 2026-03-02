// genesis-runtime/src/orchestrator/night_phase.rs
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::fs::{OpenOptions, File};
use std::io::{BufWriter, Write, Read};
use std::path::Path;
use genesis_core::constants::MAX_DENDRITE_SLOTS;

const CHKT_MAGIC: u32 = 0x43484B54; // "CHKT"

/// Выполняет прямой сброс Pinned RAM на диск без аллокаций.
unsafe fn save_hot_checkpoint(
    artifact_dir: std::path::PathBuf,
    total_ticks: u64,
    padded_n: u32,
    pinned_targets: *const u32,
    pinned_weights: *const i16,
) {
    let path = artifact_dir.join("checkpoint_weights.bin");
    let file = OpenOptions::new().create(true).write(true).truncate(true).open(&path).expect("Fatal: Failed to open checkpoint file");
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);

    writer.write_all(&CHKT_MAGIC.to_le_bytes()).unwrap();
    writer.write_all(&total_ticks.to_le_bytes()).unwrap();
    writer.write_all(&padded_n.to_le_bytes()).unwrap();

    let targets_bytes_len = (MAX_DENDRITE_SLOTS * padded_n as usize) * 4;
    let weights_bytes_len = (MAX_DENDRITE_SLOTS * padded_n as usize) * 2;

    let targets_slice = std::slice::from_raw_parts(pinned_targets as *const u8, targets_bytes_len);
    let weights_slice = std::slice::from_raw_parts(pinned_weights as *const u8, weights_bytes_len);

    writer.write_all(targets_slice).unwrap();
    writer.write_all(weights_slice).unwrap();
    writer.flush().unwrap();
}

pub unsafe fn load_hot_checkpoint(
    artifact_dir: &Path,
    expected_padded_n: u32,
    pinned_targets: *mut u32,
    pinned_weights: *mut i16,
) -> Option<u64> {
    let path = artifact_dir.join("checkpoint_weights.bin");
    if !path.exists() { return None; }

    let mut file = File::open(&path).expect("Fatal: Failed to open checkpoint");
    let mut header = [0u8; 16];
    if file.read_exact(&mut header).is_err() { return None; }

    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    assert_eq!(magic, CHKT_MAGIC);

    let tick = u64::from_le_bytes(header[4..12].try_into().unwrap());
    let padded_n = u32::from_le_bytes(header[12..16].try_into().unwrap());
    assert_eq!(padded_n, expected_padded_n);

    let targets_bytes_len = (MAX_DENDRITE_SLOTS * padded_n as usize) * 4;
    let weights_bytes_len = (MAX_DENDRITE_SLOTS * padded_n as usize) * 2;

    let targets_slice = std::slice::from_raw_parts_mut(pinned_targets as *mut u8, targets_bytes_len);
    let weights_slice = std::slice::from_raw_parts_mut(pinned_weights as *mut u8, weights_bytes_len);

    if file.read_exact(targets_slice).is_err() { return None; }
    if file.read_exact(weights_slice).is_err() { return None; }
    Some(tick)
}

pub fn trigger_night_phase(
    artifact_dir: std::path::PathBuf,
    total_ticks: u64,
    vram_ptr: *mut crate::memory::VramState, 
    padded_n: u32,
    total_axons: u32,
    prune_threshold: i16,
    is_sleeping: Arc<AtomicBool>,
    master_seed: u64,
    queues: Arc<crate::network::slow_path::SlowPathQueues>,
    inter_node_channels: Vec<crate::network::inter_node::InterNodeChannel>,
    spatial_grid: Arc<std::sync::Mutex<crate::orchestrator::spatial_grid::SpatialGrid>>,
) {
    is_sleeping.store(true, Ordering::Release);
    let vram_addr = vram_ptr as usize;

    thread::spawn(move || {
        unsafe {
            let vram = &mut *(vram_addr as *mut crate::memory::VramState);
            let stream = std::ptr::null_mut();

            // Step 0: Process Incoming Handovers & ACKs (Deferred VRAM Patching)
            // 0.1 Incoming Handovers (GROW from remote -> Local Ghost)
            let mut pending_handovers = Vec::new();
            while let Some(event) = queues.incoming_grow.pop() {
                pending_handovers.push(event);
            }

            if !pending_handovers.is_empty() {
                let acks = process_incoming_handovers(
                    vram,
                    &pending_handovers,
                );
                for ack in acks {
                    queues.outgoing_ack.push(ack);
                }
            }

            // 0.1.5 Incoming PRUNEs (Node A cut off an axon, Node B frees the slot)
            let mut pending_prunes = Vec::new();
            while let Some(prune) = queues.incoming_prune.pop() {
                pending_prunes.push(prune);
            }
            if !pending_prunes.is_empty() {
                for prune in pending_prunes {
                    vram.free_ghost_axon(prune.ghost_id);
                }
                println!("           Night Phase: FREED {} ghost axons.", queues.incoming_prune.len());
            }

            // 0.2 Received ACKs (ACK from remote for our GROW -> Patch Routing Table)
            let mut pending_acks = Vec::new();
            while let Some(ack) = queues.incoming_ack.pop() {
                pending_acks.push(ack);
            }

            if !pending_acks.is_empty() {
                patch_routing_table_during_night(
                    &inter_node_channels,
                    &pending_acks,
                    stream as crate::ffi::CudaStream,
                );
            }

            // 0.3 Find Dead Axons (Send PRUNE to remote)
            let head_index = vram.download_axon_head_index().unwrap_or_default();
            for channel in &inter_node_channels {
                for i in 0..channel.count as usize {
                    let local_id = channel.src_indices_host[i];
                    if head_index.get(local_id as usize) == Some(&0x80000000) {
                        let mut dst_ghost = vec![0u32; 1];
                        crate::ffi::gpu_memcpy_device_to_host(
                            dst_ghost.as_mut_ptr() as *mut std::ffi::c_void, 
                            channel.dst_ghost_ids_d.add(i) as *const std::ffi::c_void, 
                            4
                        );
                        if dst_ghost[0] != 0 && dst_ghost[0] != 0x80000000 {
                            queues.outgoing_prune.push(crate::network::slow_path::AxonHandoverPrune {
                                magic: 0x44454144,
                                ghost_id: dst_ghost[0],
                            });
                            println!("💀 [GC] Emitting PRUNE for dead axon: {}", dst_ghost[0]);
                            let sentinel: u32 = 0x80000000;
                            crate::ffi::gpu_memcpy_host_to_device(
                                channel.dst_ghost_ids_d.add(i) as *mut std::ffi::c_void, 
                                &sentinel as *const _ as *const std::ffi::c_void, 
                                4
                            );
                        }
                    }
                }
            }

            // Step 1: GPU Sort & Prune
            crate::ffi::launch_sort_and_prune(
                padded_n,
                vram.dendrite_targets as *mut std::ffi::c_void,
                vram.dendrite_weights as *mut std::ffi::c_void,
                vram.dendrite_refractory as *mut std::ffi::c_void,
                prune_threshold,
                stream as crate::ffi::CudaStream,
            );
            crate::ffi::gpu_stream_synchronize(stream as crate::ffi::CudaStream);

            // Step 2: Download for CPU Sprouting
            let dc = MAX_DENDRITE_SLOTS * (padded_n as usize);
            crate::ffi::gpu_memcpy_device_to_host_async(vram.pinned_host_targets as *mut _, vram.dendrite_targets, dc * 4, stream as crate::ffi::CudaStream);
            crate::ffi::gpu_memcpy_device_to_host_async(vram.pinned_host_weights as *mut _, vram.dendrite_weights, dc * 2, stream as crate::ffi::CudaStream);
            crate::ffi::gpu_stream_synchronize(stream as crate::ffi::CudaStream);

            let pinned_targets = std::slice::from_raw_parts_mut(vram.pinned_host_targets as *mut u32, dc);
            let pinned_weights = std::slice::from_raw_parts_mut(vram.pinned_host_weights as *mut i16, dc);

            // Step 3: CPU Sprouting, Nudging & Outgoing Handovers
            // 3.1 Populate SpatialGrid for candidate search
            {
                let mut grid = spatial_grid.lock().unwrap();
                grid.clear();
                // Index all active axons
                for axon_id in 0..(total_axons as usize) {
                    let packed = vram.axon_tips_uvw[axon_id];
                    if packed == 0 { continue; }
                    let tx = packed & 0x3FF;
                    let ty = (packed >> 10) & 0x3FF;
                    let tz = (packed >> 20) & 0xFF;
                    grid.insert(axon_id as u32, tx, ty, tz);
                }
            }

            let (new_synapses, handovers) = crate::orchestrator::sprouting::run_cpu_sprouting(
                pinned_targets,
                pinned_weights,
                padded_n as usize,
                total_axons,
                master_seed,
                &mut vram.axon_tips_uvw,
                &vram.axon_dirs_xyz,
                25.0, // voxel_size_um
                (7500.0, 7500.0, 2500.0), // world_size_um
                12.0, // axon_growth_step_um
                &vram.host_neuron_positions,
                &spatial_grid,
            );

            if !handovers.is_empty() {
                for h in &handovers {
                    queues.outgoing_grow.push(*h);
                }
                println!("           Night Phase: {} handovers pushed to queue.", handovers.len());
            }

            // Step 4: Upload back to VRAM
            crate::ffi::gpu_memcpy_host_to_device_async(vram.dendrite_targets, vram.pinned_host_targets, dc * 4, stream as crate::ffi::CudaStream);
            crate::ffi::gpu_memcpy_host_to_device_async(vram.dendrite_weights, vram.pinned_host_weights, dc * 2, stream as crate::ffi::CudaStream);
            crate::ffi::gpu_stream_synchronize(stream as crate::ffi::CudaStream);

            // Step 5: Hot Checkpoint Save
            save_hot_checkpoint(artifact_dir, total_ticks, padded_n, vram.pinned_host_targets as *const u32, vram.pinned_host_weights as *const i16);

            is_sleeping.store(false, Ordering::Release);
        }
    });
}

pub unsafe fn process_incoming_handovers(
    vram: &mut crate::memory::VramState,
    tcp_queue: &[crate::network::slow_path::AxonHandoverEvent],
) -> Vec<crate::network::slow_path::AxonHandoverAck> {
    let mut acks = Vec::with_capacity(tcp_queue.len());
    for event in tcp_queue {
        if let Some(ghost_id) = vram.allocate_ghost_axon() {
            // Инициализация Sentinel-ом для предотвращения ложных срабатываний
            std::ptr::write_volatile((vram.axon_head_index as *mut u32).add(ghost_id as usize), 0x80000000); 
            acks.push(crate::network::slow_path::AxonHandoverAck {
                magic: 0x41434B48,
                local_axon_id: event.local_axon_id,
                ghost_id,
            });
        } else {
            println!("⚠ Ghost Axon capacity reached. Dropping handover.");
        }
    }
    acks
}

/// [AUDIT]: Безопасное обновление таблицы маршрутизации в Night Phase.
pub unsafe fn patch_routing_table_during_night(
    channels: &[crate::network::inter_node::InterNodeChannel],
    pending_acks: &[crate::network::slow_path::AxonHandoverAck],
    stream: crate::ffi::CudaStream,
) {
    for ack in pending_acks {
        // Находим, через какой канал шел этот аксон
        for channel in channels {
            if let Some(pos) = channel.src_indices_host.iter().position(|&id| id == ack.local_axon_id) {
                let ghost_id_to_patch = ack.ghost_id; // Copy to local to avoid unaligned reference
                // Точечный патч в VRAM
                crate::ffi::gpu_memcpy_host_to_device_async(
                    channel.dst_ghost_ids_d.add(pos) as *mut _,
                    &ghost_id_to_patch as *const _ as *const _,
                    4,
                    stream
                );
            }
        }
    }
    crate::ffi::gpu_stream_synchronize(stream);
}

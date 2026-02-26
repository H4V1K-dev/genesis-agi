use crate::Runtime;
use crate::network::bsp::BspBarrier;
use crate::network::router::SpikeRouter;
use crate::network::channel::Channel;
use crate::ffi;
use std::ffi::c_void;
use crate::zone_runtime::ZoneRuntime;

pub struct DayPhase;

impl DayPhase {
    /// Runs the main GPU compute loop for one full synchronization batch across all zones.
    pub async fn run_batch(
        zones: &mut [ZoneRuntime],
        channel: &mut dyn Channel,
        barrier: &mut BspBarrier, 
        router: &mut SpikeRouter, 
        gpu_schedule_buffer: *mut c_void, 
        batch_id: u32,
        telemetry_tx: Option<&tokio::sync::broadcast::Sender<crate::network::telemetry::TelemetryPayload>>
    ) -> anyhow::Result<()> {
        let schedule = barrier.get_active_schedule();
        let batch_ticks = schedule.sync_batch_ticks;

        for current_tick in 0..batch_ticks {
            for zone in zones.iter_mut() {
                // Set the Constant Memory specifically for this zone's blueprints
                Runtime::init_constants(&zone.const_mem);

                // 0. Inject Sensory Inputs (Virtual Axons)
                if zone.runtime.vram.num_pixels > 0 {
                    unsafe {
                        let ptr = zone.runtime.vram.input_bitmask_buffer as *const c_void;
                        ffi::launch_inject_inputs(
                            zone.runtime.vram.axon_head_index,
                            ptr,
                            zone.runtime.vram.map_pixel_to_axon,
                            zone.runtime.vram.num_pixels,
                            current_tick as u32,
                            std::ptr::null_mut(),
                        );
                    }
                }

                // 1. Process Network Spikes for this specific tick
                let num_spikes = schedule.counts[current_tick];
                if num_spikes > 0 {
                    let element_offset = current_tick * 1024; // MAX_SPIKES_PER_TICK
                    let byte_offset = element_offset * std::mem::size_of::<u32>();
                    
                    unsafe {
                        let ptr = (gpu_schedule_buffer as *mut u8).add(byte_offset) as *mut c_void;
                        ffi::launch_apply_spike_batch_impl(
                            num_spikes,
                            ptr,
                            zone.runtime.vram.axon_head_index,
                            std::ptr::null_mut(),
                        );
                    }
                }

                // 2. Propagate Axons, Update Neurons, Apply GSOP
                zone.runtime.tick();

                // 3. Record Readout Interface (Batching raw spikes)
                if zone.runtime.vram.num_mapped_somas > 0 {
                    unsafe {
                        ffi::launch_record_readout(
                            zone.runtime.vram.flags,
                            zone.runtime.vram.mapped_soma_ids,
                            zone.runtime.vram.output_history,
                            zone.runtime.vram.num_mapped_somas,
                            current_tick as u32,
                            std::ptr::null_mut(),
                        );
                    }
                }
            }

            // Sync Ghost Axons between local zones
            channel.sync_spikes(zones);
        }

        // Wait for all GPU streams to finish before network barrier
        for zone in zones.iter() {
            zone.runtime.synchronize();
            
            // Extract the completed output_history batch from GPU
            if zone.runtime.vram.num_mapped_somas > 0 {
                if let Ok(history) = zone.runtime.vram.download_output_history() {
                    // TODO: Emit this byte array to the network / hub.
                    // For now, we just have the data in memory.
                    // println!("[Node] Zone {} flushed {} bytes of readout history", zone.name, history.len());
                }
            }
        }

        // Flush outbound router queues and run the BSP Barrier Sync
        let outgoing = router.flush_outgoing();
        barrier.sync_and_swap(outgoing, batch_id).await?;
        
        Ok(())
    }
}

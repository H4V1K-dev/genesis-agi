use crate::Runtime;
use crate::network::bsp::BspBarrier;
use crate::ffi;
use std::ffi::c_void;

pub struct DayPhase;

impl DayPhase {
    /// Runs the main GPU compute loop for one full synchronization batch.
    pub fn run_batch(runtime: &mut Runtime, barrier: &BspBarrier, gpu_schedule_buffer: *mut c_void) {
        let schedule = barrier.get_active_schedule();
        let batch_ticks = schedule.sync_batch_ticks;

        for current_tick in 0..batch_ticks {
            // 1. Process Network Spikes for this specific tick
            let num_spikes = schedule.counts[current_tick];
            if num_spikes > 0 {
                // The buffer is flat. Calculate offset (number of elements, not bytes!)
                let element_offset = current_tick * 1024; // MAX_SPIKES_PER_TICK
                let byte_offset = element_offset * std::mem::size_of::<u32>();
                
                unsafe {
                    let ptr = (gpu_schedule_buffer as *mut u8).add(byte_offset) as *mut c_void;
                    ffi::launch_apply_spike_batch_impl(
                        num_spikes,
                        ptr,
                        runtime.vram.axon_head_index,
                        std::ptr::null_mut(),
                    );
                }
            }

            // 2. Propagate Axons
            runtime.tick(); // Existing logic (Propagate, UpdateNeurons, ApplyGSOP)

            // 3. (Future) RecordOutgoingSpikes
            // ffi::launch_record_outputs(...)
        }

        // Wait for all GPU streams to finish before network barrier
        runtime.synchronize();
    }
}

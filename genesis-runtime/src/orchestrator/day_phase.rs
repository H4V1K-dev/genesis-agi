use std::ffi::c_void;
use crate::ffi;

pub fn execute_day_batch(
    zone: &mut crate::zone_runtime::ZoneRuntime, 
    batch_ticks: u32, 
    stream: crate::ffi::CudaStream,
    telemetry_tx: Option<&tokio::sync::broadcast::Sender<crate::network::telemetry::TelemetryPayload>>,
    total_ticks: u64,
    global_dopamine: i16
) {

    // Zero-Downtime Hot-Reload & Async Constant Memory injection
    let mut mem_updated = false;
    while let Some(new_const_mem) = zone.hot_reload_queue.pop() {
        zone.const_mem = new_const_mem;
        mem_updated = true;
    }
    if mem_updated {
        println!("🚀 [Hot-Reload] 448 bytes of GpuVariantParameters injected to VRAM __constant__!");
    }
    unsafe {
        ffi::update_global_dopamine(global_dopamine, stream);
        ffi::update_constant_memory_hot_reload(
            zone.const_mem.as_ptr(),
            stream,
        );
    }

    let layout_vram = genesis_core::layout::VramState {
        voltage: zone.runtime.vram.voltage as *mut i32,
        flags: zone.runtime.vram.flags as *mut u8,
        threshold_offset: zone.runtime.vram.threshold_offset as *mut i32,
        refractory_timer: zone.runtime.vram.refractory_timer as *mut u8,
        soma_to_axon: zone.runtime.vram.soma_to_axon as *mut u32,
        dendrite_targets: zone.runtime.vram.dendrite_targets as *mut u32,
        dendrite_weights: zone.runtime.vram.dendrite_weights as *mut i16,
        dendrite_timers: zone.runtime.vram.dendrite_refractory as *mut u8,
        axon_heads: zone.runtime.vram.axon_head_index as *mut u32,
        input_bitmask: zone.runtime.vram.input_bitmask_buffer as *mut u32,
        output_history: zone.runtime.vram.output_history as *mut u8,
        telemetry_spikes: zone.runtime.vram.telemetry_spikes as *mut u32,
        telemetry_count: zone.runtime.vram.telemetry_count as *mut u32,
    };
    let vram_ptr = &layout_vram as *const genesis_core::layout::VramState;

    unsafe { ffi::gpu_reset_telemetry_count(vram_ptr, stream); }

    let padded_n = zone.runtime.vram.padded_n as u32;
    let total_axons = zone.runtime.vram.total_axons as u32;
    
    let virtual_offset = 0u32;
    let total_virtual_axons = zone.runtime.vram.num_pixels;
    let input_stride = zone.runtime.vram.input_stride as u8;
    
    let v_seg = zone.runtime.v_seg;
    let num_output_channels = zone.runtime.vram.num_mapped_somas;
    let mapped_soma_ids = zone.runtime.vram.mapped_soma_ids as *const u32;

    let schedule_buffer = if zone.ping_pong.reading_from_a.load(std::sync::atomic::Ordering::Relaxed) {
        zone.ping_pong.buffer_a
    } else {
        zone.ping_pong.buffer_b
    };
    
    let counts_buffer = if zone.ping_pong.reading_from_a.load(std::sync::atomic::Ordering::Relaxed) {
        zone.ping_pong.counts_a
    } else {
        zone.ping_pong.counts_b
    };
    
    let max_spikes_per_tick = zone.ping_pong.max_spikes_per_tick as u32;

    for current_tick in 0..batch_ticks {
        unsafe {
            if total_virtual_axons > 0 {
                ffi::launch_inject_inputs(
                    vram_ptr, virtual_offset, current_tick as u32, input_stride, total_virtual_axons, stream
                );
            }

            let is_sleeping = zone.is_sleeping.load(std::sync::atomic::Ordering::Acquire);
            
            // Log spike buffer before applying
            let spike_count_in_buffer = unsafe { std::ptr::read_volatile(counts_buffer.add(current_tick as usize)) };
            if spike_count_in_buffer > 0 || current_tick == 0 {
                eprintln!("[Day Phase] Zone={} Tick={} is_sleeping={} spike_count={} schedule_buffer={:p}", 
                    zone.name, current_tick, is_sleeping, spike_count_in_buffer, schedule_buffer);
            }
            
            if !is_sleeping {
                if spike_count_in_buffer > 0 {
                    eprintln!("[GPU Apply] Zone={} Tick={} launching apply_spike_batch with {} spikes", zone.name, current_tick, spike_count_in_buffer);
                }
                
                ffi::launch_apply_spike_batch(vram_ptr, schedule_buffer, counts_buffer, current_tick as u32, max_spikes_per_tick, stream);
                ffi::launch_propagate_axons(vram_ptr, total_axons, v_seg, stream);
                ffi::launch_update_neurons(vram_ptr, padded_n, stream);
                ffi::launch_apply_gsop(vram_ptr, padded_n, stream);
            }

            if num_output_channels > 0 && !mapped_soma_ids.is_null() {
                eprintln!("[RecordReadout] Zone={} Tick={} num_output_channels={} mapped_soma_ids={:p}", 
                    zone.name, current_tick, num_output_channels, mapped_soma_ids);
                ffi::launch_record_readout(
                    vram_ptr, mapped_soma_ids, num_output_channels, current_tick as u32, stream
                );
            } else if num_output_channels > 0 {
                eprintln!("[RecordReadout] WARNING: num_output_channels={} but mapped_soma_ids is NULL!", num_output_channels);
            } else {
                eprintln!("[RecordReadout] SKIP: num_output_channels=0 (this is only Node B with output role)");
            }
        }
    }

    zone.runtime.synchronize();
    
    // Информационный лог: параметры выходного канала
    // DMA Device-to-Host для UDP выполняется в main.rs после gpu_stream_synchronize
    if num_output_channels > 0 {
        eprintln!(
            "[Output] Zone={} num_output_channels={}, batch_ticks={}. DMA will happen in main.rs after stream sync.",
            zone.name, num_output_channels, batch_ticks
        );
    }

    if let Some(tx) = telemetry_tx {
        let mut h_count: u32 = 0;
        unsafe {
            ffi::gpu_memcpy_device_to_host_async(
                &mut h_count as *mut _ as *mut c_void,
                zone.runtime.vram.telemetry_count as *const _,
                4,
                std::ptr::null_mut(),
            );
        }

        eprintln!("[Telemetry] Zone={} Batch output spikes: {}", zone.name, h_count);

        let safe_count = std::cmp::min(h_count, 500_000);

        if safe_count > 0 {
            unsafe {
                ffi::gpu_memcpy_device_to_host_async(
                    zone.runtime.vram.telemetry_spikes_host,
                    zone.runtime.vram.telemetry_spikes as *const _,
                    (safe_count as usize) * 4,
                    std::ptr::null_mut(),
                );
            }

            let spikes = unsafe {
                std::slice::from_raw_parts(
                    zone.runtime.vram.telemetry_spikes_host as *const u32,
                    safe_count as usize,
                ).to_vec()
            };

            let _ = tx.send(crate::network::telemetry::TelemetryPayload {
                tick: total_ticks, 
                active_spikes: spikes,
            });
        }
    }
}

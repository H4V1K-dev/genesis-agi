use crate::ffi;
use std::ffi::c_void;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use genesis_core::layout::{padded_n, StateFileHeader, AxonsFileHeader};

/// Typesafe wrapper over device pointers for the GPU SoA layout.

pub struct VramState {
    pub padded_n: usize,
    
    // Soma State
    pub voltage: *mut c_void,
    pub threshold_offset: *mut c_void,
    pub refractory_timer: *mut c_void,
    pub flags: *mut c_void,

    // Axon State (total_axons length, not padded_n)
    pub total_axons: usize,
    pub ghost_axons_allocated: usize,
    pub max_ghost_axons: usize,
    pub base_axons: usize,
    pub axon_head_index: *mut c_void,
    pub soma_to_axon: *mut c_void,

    // Dendrite Columns (MAX_DENDRITE_SLOTS * padded_n length)
    pub dendrite_targets: *mut c_void,
    pub dendrite_weights: *mut c_void,
    pub dendrite_refractory: *mut c_void,

    // Virtual Axons (InjectInputs)
    pub num_pixels: u32,
    pub map_pixel_to_axon: *mut c_void,
    pub input_bitmask_buffer: *mut c_void,

    // Outbound Spikes (Per-Tick, MAX_SPIKES_PER_TICK length) - REMOVED

    // Readout Interface (Output §3)
    pub num_mapped_somas: u32,
    pub readout_batch_ticks: u32,
    pub mapped_soma_ids: *mut c_void,   // [total_mapped_somas] u32
    pub output_history: *mut c_void,     // [batch_ticks × total_mapped_somas] u8
}

impl VramState {
    /// Loads the raw binary `.state` and `.axons` blobs from baker and 
    /// zero-copy migrates them into GPU VRAM (SoA layout).
    pub fn load_shard(state_bytes: &[u8], axons_bytes: &[u8], gxi: Option<&crate::input::GxiFile>, gxo: Option<&crate::output::GxoFile>) -> anyhow::Result<Self> {
        let axons_header = AxonsFileHeader::from_bytes(axons_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        let num_axons = axons_header.total_axons as usize;

        let pa = padded_n(num_axons);

        let state_header = StateFileHeader::from_bytes(state_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        
        let pn = state_header.padded_n as usize;
        let dc = MAX_DENDRITE_SLOTS * pn;

        let mut offset = state_header.header_size as usize;
        // --- Inject Inputs Context ---
        let mut map_pixel_to_axon = std::ptr::null_mut();
        let mut num_pixels = 0;
        let mut bitmask_buffer = std::ptr::null_mut();
        
        let batch_size_ticks = 1000; // max batch ticks

        if let Some(g) = gxi {
            num_pixels = g.axon_ids.len() as u32;
            if num_pixels > 0 {
                unsafe {
                    let bytes = (num_pixels as usize) * 4;
                    map_pixel_to_axon = ffi::gpu_malloc(bytes);
                    if map_pixel_to_axon.is_null() {
                        anyhow::bail!("gpu_malloc failed for map_pixel_to_axon ({} bytes)", bytes);
                    }
                    let success = ffi::gpu_memcpy_host_to_device(
                        map_pixel_to_axon,
                        g.axon_ids.as_ptr() as *const c_void,
                        bytes,
                    );
                    if !success {
                        anyhow::bail!("Failed to upload map_pixel_to_axon to GPU");
                    }

                    let bitmask_bytes = ((num_pixels as usize + 31) / 32) * 4 * batch_size_ticks;
                    bitmask_buffer = ffi::gpu_malloc(bitmask_bytes);
                    if bitmask_buffer.is_null() {
                        anyhow::bail!("gpu_malloc failed for input_bitmask_buffer ({} bytes)", bitmask_bytes);
                    }
                }
            }
        }
        let mut allocate_and_copy = |slice_len: usize| -> anyhow::Result<*mut c_void> {
            let ptr = unsafe { ffi::gpu_malloc(slice_len) };
            if ptr.is_null() {
                anyhow::bail!("gpu_malloc failed for size {}", slice_len);
            }
            let success = unsafe {
                ffi::gpu_memcpy_host_to_device(
                    ptr,
                    state_bytes[offset..offset + slice_len].as_ptr() as *const c_void,
                    slice_len,
                )
            };
            if !success {
                anyhow::bail!("gpu_memcpy_host_to_device failed for size {}", slice_len);
            }
            offset += slice_len;
            Ok(ptr)
        };

        let voltage = allocate_and_copy(pn * 4)?;
        let flags = allocate_and_copy(pn * 1)?;
        let threshold_offset = allocate_and_copy(pn * 4)?;
        let refractory_timer = allocate_and_copy(pn * 1)?;
        let soma_to_axon = allocate_and_copy(pn * 4)?;
        let dendrite_targets = allocate_and_copy(dc * 4)?;
        let dendrite_weights = allocate_and_copy(dc * 2)?;
        let dendrite_refractory = allocate_and_copy(dc * 1)?;
        
        // Axon Heads: Base + Pre-allocate 10000 Ghost Axons
        let max_ghost_axons = 10000;
        let total_axons = pa + max_ghost_axons;
        let axon_head_index = unsafe { ffi::gpu_malloc(total_axons * 4) };
        if axon_head_index.is_null() { anyhow::bail!("alloc failed for axon heads"); }
        
        // Copy base axons
        unsafe {
            ffi::gpu_memcpy_host_to_device(
                axon_head_index,
                state_bytes[offset..offset + pa * 4].as_ptr() as *const c_void,
                pa * 4,
            );
        }
        offset += pa * 4;

        // Init spare Ghost Axons to AXON_SENTINEL
        let sentinels = vec![0x80000000u32; max_ghost_axons];
        unsafe {
            ffi::gpu_memcpy_host_to_device(
                (axon_head_index as *mut u32).add(pa) as *mut c_void,
                sentinels.as_ptr() as *const c_void,
                max_ghost_axons * 4,
            );
        }

        // Readout Buffer Allocation
        let mut mapped_soma_ids = std::ptr::null_mut();
        let mut output_history = std::ptr::null_mut();
        let mut num_mapped_somas = 0;
        let mut readout_batch_ticks = 0;

        if let Some(o) = gxo {
            num_mapped_somas = o.total_somas;
            readout_batch_ticks = o.readout_batch_ticks;
            if num_mapped_somas > 0 && readout_batch_ticks > 0 {
                unsafe {
                    let somas_bytes = (num_mapped_somas as usize) * 4;
                    mapped_soma_ids = ffi::gpu_malloc(somas_bytes);
                    if mapped_soma_ids.is_null() {
                        anyhow::bail!("gpu_malloc failed for mapped_soma_ids ({} bytes)", somas_bytes);
                    }
                    if !ffi::gpu_memcpy_host_to_device(mapped_soma_ids, o.mapped_soma_ids.as_ptr() as *const c_void, somas_bytes) {
                        anyhow::bail!("Failed to upload mapped_soma_ids to GPU");
                    }

                    // output_history buffer (u8 per tick per soma)
                    let history_bytes = (num_mapped_somas as usize) * (readout_batch_ticks as usize);
                    output_history = ffi::gpu_malloc(history_bytes);
                    if output_history.is_null() {
                        anyhow::bail!("gpu_malloc failed for output_history ({} bytes)", history_bytes);
                    }
                    // It's good practice to zero it out, though the kernel writes absolutely every byte
                    // we'll leave it uninitialized on GPU to save time, it will be fully overwritten over the batch ticks.
                }
            }
        }



        Ok(VramState {
            padded_n: pn,
            total_axons,
            ghost_axons_allocated: 0,
            max_ghost_axons,
            base_axons: pa,
            num_pixels,
            map_pixel_to_axon,
            input_bitmask_buffer: bitmask_buffer,
            voltage,
            threshold_offset,
            refractory_timer,
            flags,
            soma_to_axon,
            axon_head_index,
            dendrite_targets,
            dendrite_weights,
            dendrite_refractory,
            num_mapped_somas,
            readout_batch_ticks,
            mapped_soma_ids,
            output_history,
        })
    }

    /// Extracted helper to launch the Night Phase Sort & Prune kernel
    pub fn run_sort_and_prune(&self, prune_threshold: i16) {
        unsafe {
            ffi::launch_sort_and_prune(
                self.padded_n as u32,
                self.dendrite_targets,
                self.dendrite_weights,
                self.dendrite_refractory, // Using timers
                prune_threshold,
                std::ptr::null_mut()
            );
        }
    }

    /// Downloads a generic slice of data from the GPU.
    fn download_generic<T: Clone + Default>(&self, ptr: *mut c_void, count: usize) -> anyhow::Result<Vec<T>> {
        let size = count * std::mem::size_of::<T>();
        let mut host_data = vec![T::default(); count];
        
        let success = unsafe {
            ffi::gpu_memcpy_device_to_host(
                host_data.as_mut_ptr() as *mut c_void,
                ptr as *const c_void,
                size,
            )
        };

        if !success {
            anyhow::bail!("gpu_memcpy_device_to_host failed for size {}", size);
        }

        Ok(host_data)
    }

    pub fn download_voltage(&self) -> anyhow::Result<Vec<i32>> {
        self.download_generic(self.voltage, self.padded_n)
    }

    pub fn download_flags(&self) -> anyhow::Result<Vec<u8>> {
        self.download_generic(self.flags, self.padded_n)
    }

    pub fn download_threshold_offset(&self) -> anyhow::Result<Vec<i32>> {
        self.download_generic(self.threshold_offset, self.padded_n)
    }

    pub fn download_refractory_timer(&self) -> anyhow::Result<Vec<u8>> {
        self.download_generic(self.refractory_timer, self.padded_n)
    }

    pub fn download_axon_head_index(&self) -> anyhow::Result<Vec<u32>> {
        self.download_generic(self.axon_head_index, self.total_axons)
    }

    pub fn download_dendrite_weights(&self) -> anyhow::Result<Vec<i16>> {
        self.download_generic(self.dendrite_weights, self.padded_n * MAX_DENDRITE_SLOTS)
    }

    pub fn download_dendrite_targets(&self) -> anyhow::Result<Vec<u32>> {
        self.download_generic(self.dendrite_targets, self.padded_n * MAX_DENDRITE_SLOTS)
    }

    pub fn upload_dendrite_weights(&self, host_data: &[i16]) -> anyhow::Result<()> {
        let expected_len = self.padded_n * MAX_DENDRITE_SLOTS;
        if host_data.len() != expected_len {
            anyhow::bail!("Invalid length: expected {}, got {}", expected_len, host_data.len());
        }
        let size = expected_len * std::mem::size_of::<i16>();
        let success = unsafe {
            ffi::gpu_memcpy_host_to_device(
                self.dendrite_weights,
                host_data.as_ptr() as *const std::ffi::c_void,
                size,
            )
        };
        if !success {
            anyhow::bail!("gpu_memcpy_host_to_device failed for dendrite weights");
        }
        Ok(())
    }

    pub fn upload_dendrite_targets(&self, host_data: &[u32]) -> anyhow::Result<()> {
        let expected_len = self.padded_n * MAX_DENDRITE_SLOTS;
        if host_data.len() != expected_len {
            anyhow::bail!("Invalid length: expected {}, got {}", expected_len, host_data.len());
        }
        let size = expected_len * std::mem::size_of::<u32>();
        let success = unsafe {
            ffi::gpu_memcpy_host_to_device(
                self.dendrite_targets,
                host_data.as_ptr() as *const std::ffi::c_void,
                size,
            )
        };
        if !success {
            anyhow::bail!("gpu_memcpy_host_to_device failed for dendrite targets");
        }
        Ok(())
    }

    pub fn download_dendrite_timers(&self) -> anyhow::Result<Vec<u8>> {
        self.download_generic(self.dendrite_refractory, self.padded_n * MAX_DENDRITE_SLOTS)
    }

    pub fn download_output_history(&self) -> anyhow::Result<Vec<u8>> {
        if self.num_mapped_somas == 0 || self.readout_batch_ticks == 0 {
            return Ok(Vec::new());
        }
        let total_bytes = (self.num_mapped_somas as usize) * (self.readout_batch_ticks as usize);
        self.download_generic(self.output_history, total_bytes)
    }

    /// Uploads a bitmask array to GPU memory. Used for External Virtual Axons.
    /// Bitmask must be accurately sized: ((num_pixels + 31)/32) u32s times batch size.
    pub fn upload_input_bitmask(&self, bitmask: &[u32], num_ticks: usize) -> anyhow::Result<()> {
        if self.num_pixels == 0 {
            return Ok(());
        }
        let max_ticks = 1000;
        if num_ticks > max_ticks {
            anyhow::bail!("Batch size too large: {} (max {})", num_ticks, max_ticks);
        }
        
        let words_per_tick = (self.num_pixels as usize + 31) / 32;
        let total_words = words_per_tick * num_ticks;
        
        if bitmask.len() < total_words {
            anyhow::bail!("Bitmask len {} is less than required {} for {} ticks", bitmask.len(), total_words, num_ticks);
        }

        let bytes = total_words * std::mem::size_of::<u32>();
        let success = unsafe {
            ffi::gpu_memcpy_host_to_device(
                self.input_bitmask_buffer,
                bitmask.as_ptr() as *const c_void,
                bytes,
            )
        };
        if !success {
            anyhow::bail!("Failed to upload input bitmask to GPU");
        }

        Ok(())
    }

    pub fn allocate_ghost_axon(&mut self) -> Option<u32> {
        if self.ghost_axons_allocated < self.max_ghost_axons {
            let id = (self.base_axons + self.ghost_axons_allocated) as u32;
            self.ghost_axons_allocated += 1;
            Some(id)
        } else {
            None
        }
    }

    pub fn free_ghost_axon(&mut self, ghost_id: u32) {
        if (ghost_id as usize) >= self.base_axons && (ghost_id as usize) < self.base_axons + self.ghost_axons_allocated {
            let sentinel: u32 = 0x80000000;
            let offset = ghost_id as usize;
            unsafe {
                ffi::gpu_memcpy_host_to_device(
                    (self.axon_head_index as *mut u32).add(offset) as *mut std::ffi::c_void,
                    &sentinel as *const _ as *const std::ffi::c_void,
                    4,
                );
            }
        }
    }
}

impl Drop for VramState {
    fn drop(&mut self) {
        unsafe {
            if !self.voltage.is_null() { ffi::gpu_free(self.voltage); }
            if !self.threshold_offset.is_null() { ffi::gpu_free(self.threshold_offset); }
            if !self.refractory_timer.is_null() { ffi::gpu_free(self.refractory_timer); }
            if !self.flags.is_null() { ffi::gpu_free(self.flags); }

            if !self.axon_head_index.is_null() { ffi::gpu_free(self.axon_head_index); }
            if !self.soma_to_axon.is_null() { ffi::gpu_free(self.soma_to_axon); }

            if !self.dendrite_targets.is_null() { ffi::gpu_free(self.dendrite_targets); }
            if !self.dendrite_weights.is_null() { ffi::gpu_free(self.dendrite_weights); }
            if !self.dendrite_refractory.is_null() { ffi::gpu_free(self.dendrite_refractory); }

            if !self.mapped_soma_ids.is_null() {
                ffi::gpu_free(self.mapped_soma_ids);
            }
            if !self.output_history.is_null() {
                ffi::gpu_free(self.output_history);
            }

            if !self.map_pixel_to_axon.is_null() {
                ffi::gpu_free(self.map_pixel_to_axon);
            }
            if !self.input_bitmask_buffer.is_null() {
                ffi::gpu_free(self.input_bitmask_buffer);
            }
        }
    }
}

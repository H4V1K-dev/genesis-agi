pub mod ffi;
pub mod memory;
pub mod network;
pub mod orchestrator;

use memory::VramState;
use std::ptr;
use std::ffi::c_void;

#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
pub struct VariantParameters {
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak: i32,
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: i32,
    pub gsop_potentiation: u16,
    pub gsop_depression: u16,
    pub refractory_period: u8,
    pub synapse_refractory: u8,
    pub slot_decay_ltm: u8,
    pub slot_decay_wm: u8,
    pub _padding: [u8; 4],
}

#[repr(C, align(128))]
#[derive(Clone, Copy)]
pub struct GenesisConstantMemory {
    pub variants: [VariantParameters; 4],
    pub inertia_lut: [u8; 16],
    pub _padding: [u8; 112],
}

impl Default for GenesisConstantMemory {
    fn default() -> Self {
        Self {
            variants: [VariantParameters::default(); 4],
            inertia_lut: [0; 16],
            _padding: [0; 112],
        }
    }
}

pub struct Runtime {
    pub vram: VramState,
    pub v_seg: u32,
}

impl Runtime {
    pub fn new(vram: VramState, v_seg: u32) -> Self {
        Self { vram, v_seg }
    }

    pub fn init_constants(constants: &GenesisConstantMemory) -> bool {
        unsafe { ffi::upload_constant_memory(constants as *const _ as *const c_void) }
    }

    /// Executed on the GPU every engine tick (Day Phase).
    pub fn tick(&mut self) {
        unsafe {
            // 1. Propagate Axons
            ffi::launch_propagate_axons(
                self.vram.total_axons as u32,
                self.vram.axon_head_index,
                self.v_seg,
                ptr::null_mut(),
            );

            // 2. Update Neurons
            ffi::launch_update_neurons(
                self.vram.padded_n as u32,
                self.vram.voltage,
                self.vram.threshold_offset,
                self.vram.refractory_timer,
                self.vram.flags,
                self.vram.soma_to_axon,
                self.vram.dendrite_targets,
                self.vram.dendrite_weights,
                self.vram.axon_head_index,
                ptr::null_mut(),
            );

            // 3. Apply GSOP
            ffi::launch_apply_gsop(
                self.vram.padded_n as u32,
                self.vram.flags,
                self.vram.dendrite_targets,
                self.vram.dendrite_weights,
                self.vram.dendrite_refractory, // Note: passing refractory as timers
                self.vram.axon_head_index,
                ptr::null_mut(),
            );
        }
    }

    pub fn synchronize(&self) {
        unsafe { ffi::gpu_device_synchronize() };
    }
}

use crate::types::{Voltage, Weight};

pub const MAX_DENDRITES: usize = 128;

/// Структура параметров типа нейрона.
/// Выровнена по 32 байта для GPU Constant Memory (16 типов = 512 байт).
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct VariantParameters {
    pub threshold: i32,                 // 4B
    pub rest_potential: i32,            // 4B
    pub leak_rate: i32,                 // 4B
    pub homeostasis_penalty: i32,       // 4B
    pub homeostasis_decay: u16,         // 2B
    pub gsop_potentiation: i16,         // 2B
    pub gsop_depression: i16,           // 2B
    pub refractory_period: u8,          // 1B
    pub synapse_refractory_period: u8,  // 1B
    pub slot_decay_ltm: u8,             // 1B
    pub slot_decay_wm: u8,              // 1B
    pub signal_propagation_length: u8,  // 1B
    pub _padding: [u8; 5],              // 5B -> Total: 32 Bytes
}

const _: () = assert!(std::mem::size_of::<VariantParameters>() == 32);

/// Host-side SoA state of a shard.
/// Used for baking and disk I/O.
#[repr(C)]
pub struct ShardStateSoA {
    pub padded_n: usize, // Must be multiple of 32

    // --- Soma Hot State ---
    pub voltage: Vec<Voltage>,
    pub flags: Vec<u8>,
    pub threshold_offset: Vec<i32>,
    pub refractory_timer: Vec<u8>,

    // --- Columnar Dendrites (Size = MAX_DENDRITES * padded_n) ---
    pub dendrite_targets: Vec<u32>, // Dense ID + Segment Offset
    pub dendrite_weights: Vec<Weight>,
    pub dendrite_timers: Vec<u8>,

    // --- Axon Heads ---
    pub axon_heads: Vec<u32>, 
}

impl ShardStateSoA {
    /// Вычисляет плоский индекс для Coalesced Access на GPU
    #[inline(always)]
    pub fn columnar_idx(padded_n: usize, neuron_idx: usize, slot: usize) -> usize {
        debug_assert!(neuron_idx < padded_n && slot < MAX_DENDRITES);
        slot * padded_n + neuron_idx
    }
}

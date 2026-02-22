pub mod ring_buffer;
pub mod bsp;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpikeBatchHeader {
    pub batch_id: u32,
    pub spikes_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpikeEvent {
    pub receiver_ghost_id: u32,
    pub tick_offset: u8,
    pub _pad: [u8; 3], // align to 64 bits (8 bytes)
}

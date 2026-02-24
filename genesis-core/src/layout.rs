use crate::constants::GPU_WARP_SIZE;

/// Align neuron count to warp boundary for Coalesced GPU Access.
/// All SoA arrays must use padded_n as their stride.
///
/// Columnar Layout: array[slot * padded_n + neuron_id]
pub fn padded_n(neuron_count: usize) -> usize {
    let r = neuron_count % GPU_WARP_SIZE;
    if r == 0 {
        neuron_count
    } else {
        neuron_count + (GPU_WARP_SIZE - r)
    }
}

pub const STATE_MAGIC: u32 = 0x47534E53; // "GSNS"
pub const AXONS_MAGIC: u32 = 0x47534158; // "GSAX"
pub const STATE_VERSION: u16 = 1;

/// Заголовок бинарного файла .state (16 байт)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateFileHeader {
    pub magic: u32,
    pub version: u16,
    pub header_size: u16, // = 16
    pub padded_n: u32,
    pub total_base_axons: u32,
}

impl StateFileHeader {
    pub fn new(padded_n: u32, total_base_axons: u32) -> Self {
        Self {
            magic: STATE_MAGIC,
            version: STATE_VERSION,
            header_size: 16,
            padded_n,
            total_base_axons,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    pub fn from_bytes(data: &[u8]) -> Result<&Self, &'static str> {
        if data.len() < std::mem::size_of::<Self>() {
            return Err("Data too short for StateFileHeader");
        }
        let header = unsafe { &*(data.as_ptr() as *const Self) };
        if header.magic != STATE_MAGIC {
            return Err("Invalid STATE_MAGIC (expected 'GSNS')");
        }
        if header.version != STATE_VERSION {
            return Err("Unsupported StateFile VERSION");
        }
        Ok(header)
    }
}

/// Заголовок бинарного файла .axons (8 байт)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxonsFileHeader {
    pub magic: u32,
    pub total_axons: u32,
}

impl AxonsFileHeader {
    pub fn new(total_axons: u32) -> Self {
        Self {
            magic: AXONS_MAGIC,
            total_axons,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    pub fn from_bytes(data: &[u8]) -> Result<&Self, &'static str> {
        if data.len() < std::mem::size_of::<Self>() {
            return Err("Data too short for AxonsFileHeader");
        }
        let header = unsafe { &*(data.as_ptr() as *const Self) };
        if header.magic != AXONS_MAGIC {
            return Err("Invalid AXONS_MAGIC (expected 'GSAX')");
        }
        Ok(header)
    }
}

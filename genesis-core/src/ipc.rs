/// Night Phase IPC — Shared Memory layout between genesis-runtime and
/// genesis-baker-daemon.
///
/// SHM name: `/genesis_shard_{zone_id}`
/// Layout:
///   [0..64)   ShmHeader  (fixed, repr C, 64 bytes)
///   [64..)    weights: i16 × 128 × padded_n  (little-endian)
///             targets: u32 × 128 × padded_n  (little-endian)
///
/// State machine (single-writer invariant):
///   IDLE       → runtime writes              → NIGHT_START
///   NIGHT_START → daemon reads & begins work  → SPROUTING
///   SPROUTING  → daemon writes result         → NIGHT_DONE
///   NIGHT_DONE → runtime reads & resets       → IDLE
///   Any state  → daemon panics               → ERROR

/// Magic number at offset 0 of every SHM segment.
pub const SHM_MAGIC: u32 = 0x47454E53; // "GENS"

/// IPC protocol version. Bump on incompatible ShmHeader changes.
pub const SHM_VERSION: u8 = 1;

/// Header at the very start of the SHM segment.
/// MUST remain exactly 64 bytes. Verified by `shm_header_size_assert`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmHeader {
    /// Magic: 0x47454E53
    pub magic: u32,
    /// IPC protocol version (SHM_VERSION)
    pub version: u8,
    /// Current state of the Night Phase pipeline (ShmState)
    pub state: u8,
    /// Shard zone ID
    pub zone_id: u16,
    /// Number of neurons (padded to warp boundary). Used to compute offsets.
    pub padded_n: u32,
    /// Number of dendrite slots per neuron (always 128, here for sanity check)
    pub dendrite_slots: u32,
    /// Byte offset from start of SHM to weights array
    pub weights_offset: u32,
    /// Byte offset from start of SHM to targets array
    pub targets_offset: u32,
    /// Monotonic counter incremented each Night Phase (for debugging)
    pub epoch: u64,
    pub _padding: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<ShmHeader>() == 64, "ShmHeader must be 64 bytes");

impl ShmHeader {
    /// Construct a valid header for a new SHM segment.
    pub fn new(zone_id: u16, padded_n: u32) -> Self {
        let weights_offset = std::mem::size_of::<ShmHeader>() as u32;
        let weights_bytes = padded_n * 128 * std::mem::size_of::<i16>() as u32;
        let targets_offset = weights_offset + weights_bytes;
        Self {
            magic: SHM_MAGIC,
            version: SHM_VERSION,
            state: ShmState::Idle as u8,
            zone_id,
            padded_n,
            dendrite_slots: 128,
            weights_offset,
            targets_offset,
            epoch: 0,
            _padding: [0; 32],
        }
    }

    /// Validate a header read from shared memory.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.magic != SHM_MAGIC {
            return Err("SHM magic mismatch");
        }
        if self.version != SHM_VERSION {
            return Err("SHM version mismatch");
        }
        if self.dendrite_slots != 128 {
            return Err("SHM dendrite_slots != 128");
        }
        Ok(())
    }
}

/// State machine for the SHM Night Phase protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShmState {
    /// SHM ready. Daemon waiting. Runtime may start Night Phase.
    Idle = 0,
    /// Runtime wrote weights+targets. Daemon should start Sprouting.
    NightStart = 1,
    /// Daemon is running Sprouting. Do not touch SHM data.
    Sprouting = 2,
    /// Daemon finished. Updated targets ready for runtime to read.
    NightDone = 3,
    /// Daemon encountered an error. Runtime should skip this night.
    Error = 4,
}

impl ShmState {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Idle),
            1 => Some(Self::NightStart),
            2 => Some(Self::Sprouting),
            3 => Some(Self::NightDone),
            4 => Some(Self::Error),
            _ => None,
        }
    }
}

/// Total SHM segment size in bytes for a given padded neuron count.
///
/// Layout: header (64B) + weights (i16 × 128 × N) + targets (u32 × 128 × N)
pub fn shm_size(padded_n: usize) -> usize {
    std::mem::size_of::<ShmHeader>()
        + padded_n * 128 * std::mem::size_of::<i16>()
        + padded_n * 128 * std::mem::size_of::<u32>()
}

/// Canonical POSIX SHM name for a given zone.
/// Example: zone_id=4 → "/genesis_shard_4"
pub fn shm_name(zone_id: u16) -> String {
    format!("/genesis_shard_{zone_id}")
}

/// Default Unix socket path for baker daemon control channel.
pub fn default_socket_path(zone_id: u16) -> String {
    format!("/tmp/genesis_baker_{zone_id}.sock")
}

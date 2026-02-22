/// Packed voxel coordinate: [Type(4b) | Z(8b) | Y(10b) | X(10b)]
/// Bit layout: t << 28 | z << 20 | y << 10 | x
pub type PackedPosition = u32;

/// Dendrite target: upper 16 bits = axon_id, lower 16 bits = segment offset.
/// target = 0 means empty slot.
pub type PackedTarget = u32;

/// Synaptic weight. Sign encodes excitatory (+) or inhibitory (-).
/// Range: -32768..+32767. Baked in during Night Phase, frozen during Day Phase.
pub type Weight = i16;

/// Neuron membrane voltage accumulator.
pub type Voltage = i32;

/// Axon head position (segment index). AXON_SENTINEL when inactive.
pub type AxonHead = u32;

/// Variant ID (2 bits in flags byte: bits 6-7).
/// 0..3 → index into VariantParameters[4] in Constant Memory.
pub type VariantId = u8;

/// Neuron flags byte:
/// [7:6] variant_id | [5] is_spiking | [4] reserved | [3:0] type_mask
pub type NeuronFlags = u8;

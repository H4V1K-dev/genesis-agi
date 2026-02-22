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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padded_n_aligns_to_warp() {
        assert_eq!(padded_n(32), 32);
        assert_eq!(padded_n(33), 64);
        assert_eq!(padded_n(100), 128);
        assert_eq!(padded_n(0), 0);
    }
}

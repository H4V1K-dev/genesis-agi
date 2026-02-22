use crate::bake::seed::random_f32;
use crate::parser::blueprints::NeuronType;

/// Весовая конфигурация для sprouting score (из blueprints.toml).
pub struct SproutingWeights {
    pub weight_distance: f32,
    pub weight_power: f32,
    pub weight_explore: f32,
}

impl SproutingWeights {
    pub fn from_neuron_type(nt: &NeuronType) -> Self {
        Self {
            weight_distance: nt.sprouting_weight_distance,
            weight_power: nt.sprouting_weight_power,
            weight_explore: nt.sprouting_weight_explore,
        }
    }
}

/// Нормализованный «вес» сомы = Σ|dendrite_weights| / (128 × 32767).
/// При первом Baking все веса = 0 → power_index = 0.0 (новые нейроны равны).
/// (04_connectivity.md §1.6.1)
#[allow(dead_code)]
pub fn compute_power_index(soma_id: usize, weights: &[i16], padded_n: usize) -> f32 {
    let mut power = 0u32;
    for slot in 0..128 {
        let w = weights[slot * padded_n + soma_id];
        power += w.unsigned_abs() as u32;
    }
    power as f32 / (128.0 * 32767.0)
}

/// Скоринг аксона-кандидата для данного дендрита.
/// Выше = лучше. Берётся top-N по score.
/// (04_connectivity.md §1.6.1)
pub fn sprouting_score(
    dist: f32,
    target_power: f32,
    epoch_seed: u64,
    cfg: &SproutingWeights,
) -> f32 {
    let dist_score = 1.0 / (dist + 1.0); // ближе → выше
    let power_score = target_power; // мощные сомы притягивают
    let explore = random_f32(epoch_seed); // ротация предпочтений

    dist_score * cfg.weight_distance + power_score * cfg.weight_power + explore * cfg.weight_explore
}

/// Евклидово расстояние в вокселях между двумя точками.
pub fn voxel_dist(ax: u32, ay: u32, az: u32, bx: u32, by: u32, bz: u32) -> f32 {
    let dx = ax as f32 - bx as f32;
    let dy = ay as f32 - by as f32;
    let dz = az as f32 - bz as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::blueprints;

    const BP: &str = r#"
[[neuron_type]]
name = "Vertical_Excitatory"
threshold = 42000
rest_potential = 10000
leak_rate = 1200
refractory_period = 15
synapse_refractory_period = 15
conduction_velocity = 200
signal_propagation_length = 10
axon_growth_step = 12
homeostasis_penalty = 5000
homeostasis_decay = 10
slot_decay_ltm = 160
slot_decay_wm = 96
sprouting_weight_distance = 0.5
sprouting_weight_power = 0.4
sprouting_weight_explore = 0.1
"#;

    #[test]
    fn compute_power_index_zero_weights() {
        let weights = vec![0i16; 128 * 32];
        let pi = compute_power_index(0, &weights, 32);
        assert_eq!(pi, 0.0, "blank shard: all power indices must be 0.0");
    }

    #[test]
    fn compute_power_index_max_weights() {
        let weights = vec![i16::MAX; 128 * 32];
        let pi = compute_power_index(0, &weights, 32);
        assert!(
            (pi - 1.0).abs() < 1e-4,
            "max weights → power_index ≈ 1.0, got {}",
            pi
        );
    }

    #[test]
    fn sprouting_score_closer_wins() {
        let bp = blueprints::parse(BP).unwrap();
        let cfg = SproutingWeights::from_neuron_type(&bp.neuron_type[0]);
        let score_near = sprouting_score(1.0, 0.5, 42, &cfg);
        let score_far = sprouting_score(100.0, 0.5, 42, &cfg);
        assert!(score_near > score_far, "closer axon must score higher");
    }

    #[test]
    fn voxel_dist_zero_same_point() {
        assert_eq!(voxel_dist(5, 5, 5, 5, 5, 5), 0.0);
    }
}

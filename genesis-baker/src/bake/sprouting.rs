use crate::bake::seed::random_f32;
use crate::parser::blueprints::NeuronType;

/// Весовая конфигурация для sprouting score (из blueprints.toml).
pub struct SproutingWeights {
    pub weight_distance: f32,
    pub weight_power: f32,
    pub weight_explore: f32,
    pub weight_type: f32,
}

impl SproutingWeights {
    pub fn from_neuron_type(nt: &NeuronType) -> Self {
        Self {
            weight_distance: nt.sprouting_weight_distance,
            weight_power: nt.sprouting_weight_power,
            weight_explore: nt.sprouting_weight_explore,
            weight_type: nt.sprouting_weight_type,
        }
    }
}

/// Нормализованный «вес» сомы = Σ|dendrite_weights| / (128 × 32767).
/// При первом Baking все веса = 0 → power_index = 0.0 (новые нейроны равны).
/// (04_connectivity.md §1.6.1)
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
    type_affinity: f32,
    is_same_type: bool,
) -> f32 {
    let dist_score = 1.0 / (dist + 1.0); // ближе → выше
    let power_score = target_power; // мощные сомы притягивают
    let explore = random_f32(epoch_seed); // ротация предпочтений
    let type_score = if is_same_type { type_affinity } else { 1.0 - type_affinity };

    dist_score * cfg.weight_distance 
        + power_score * cfg.weight_power 
        + explore * cfg.weight_explore
        + type_score * cfg.weight_type
}

/// Евклидово расстояние в вокселях между двумя точками.
pub fn voxel_dist(ax: u32, ay: u32, az: u32, bx: u32, by: u32, bz: u32) -> f32 {
    let dx = ax as f32 - bx as f32;
    let dy = ay as f32 - by as f32;
    let dz = az as f32 - bz as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}


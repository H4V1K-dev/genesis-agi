use crate::bake::axon_growth::GrownAxon;
use crate::bake::layout::ShardStateSoA;
use crate::bake::neuron_placement::PlacedNeuron;
use crate::bake::seed::entity_seed;
use crate::bake::sprouting::{sprouting_score, voxel_dist, SproutingWeights};
use crate::parser::blueprints::NeuronType;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use std::collections::HashMap;

pub const INITIAL_EXCITATORY_WEIGHT: i16 = 74;
pub const INITIAL_INHIBITORY_WEIGHT: i16 = -74;

/// Размер ячейки пространственной решётки (в вокселях).
/// Кандидат может находиться не дальше CELL_SIZE от сомы → ищем только в 3×3×3 ячейках.
const CELL_SIZE: u32 = 30;

/// Ключ ячейки пространственной решётки.
type GridCell = (u32, u32, u32);

/// Строит HashMap: grid_cell → список индексов аксонов с tip в этой ячейке.
/// Позволяет заменить O(N²) полный перебор на O(N × K) поиск по соседям.
fn build_axon_grid(axons: &[GrownAxon]) -> HashMap<GridCell, Vec<usize>> {
    let mut grid: HashMap<GridCell, Vec<usize>> = HashMap::new();
    for (i, ax) in axons.iter().enumerate() {
        let cell = (
            ax.tip_x / CELL_SIZE,
            ax.tip_y / CELL_SIZE,
            ax.tip_z / CELL_SIZE,
        );
        grid.entry(cell).or_default().push(i);
    }
    grid
}

/// Кандидат дендритного слота.
struct Candidate {
    axon_idx: usize,
    score: f32,
}

/// Заполняет dendrite_targets и dendrite_weights в ShardStateSoA.
/// Использует пространственную решётку для O(N × K) поиска вместо O(N²).
pub fn connect_dendrites(
    shard: &mut ShardStateSoA,
    neurons: &[PlacedNeuron],
    axons: &[GrownAxon],
    neuron_types: &[NeuronType],
    master_seed: u64,
) {
    let pn = shard.padded_n;

    // Строим пространственную решётку по tip-позициям аксонов
    let grid = build_axon_grid(axons);

    for (soma_id, neuron) in neurons.iter().enumerate() {
        let soma_x = neuron.x();
        let soma_y = neuron.y();
        let soma_z = neuron.z();
        let nt = &neuron_types[neuron.type_idx.min(neuron_types.len() - 1)];
        let cfg = SproutingWeights::from_neuron_type(nt);

        // Диапазон ячеек для поиска
        let cell_x = soma_x / CELL_SIZE;
        let cell_y = soma_y / CELL_SIZE;
        let cell_z = soma_z / CELL_SIZE;

        let mut candidates: Vec<Candidate> = Vec::new();

        // Проверяем 3×3×3 соседних ячейки
        for dx in 0..=2u32 {
            for dy in 0..=2u32 {
                for dz in 0..=2u32 {
                    let cx = cell_x.saturating_add(dx).saturating_sub(1);
                    let cy = cell_y.saturating_add(dy).saturating_sub(1);
                    let cz = cell_z.saturating_add(dz).saturating_sub(1);

                    let Some(cell_axons) = grid.get(&(cx, cy, cz)) else {
                        continue;
                    };

                    for &axon_idx in cell_axons {
                        let ax = &axons[axon_idx];
                        if ax.soma_idx == soma_id {
                            continue;
                        }

                        let dist = voxel_dist(soma_x, soma_y, soma_z, ax.tip_x, ax.tip_y, ax.tip_z);
                        if dist > CELL_SIZE as f32 {
                            continue;
                        }

                        let epoch_seed = entity_seed(
                            master_seed,
                            (soma_id.wrapping_mul(31).wrapping_add(axon_idx)) as u32,
                        );
                        let score = sprouting_score(dist, 0.0, epoch_seed, &cfg);
                        candidates.push(Candidate { axon_idx, score });
                    }
                }
            }
        }

        // Сортируем по убыванию score
        candidates.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Записываем top-N в columnar SoA
        for (slot, cand) in candidates.iter().take(MAX_DENDRITE_SLOTS).enumerate() {
            let axon_idx = cand.axon_idx;
            let target_type = axons[axon_idx].type_idx;
            let is_excitatory = neuron_types
                .get(target_type)
                .map(|n| n.name.contains("Excitatory"))
                .unwrap_or(true);
            let weight = if is_excitatory {
                INITIAL_EXCITATORY_WEIGHT
            } else {
                INITIAL_INHIBITORY_WEIGHT
            };

            let cell = slot * pn + soma_id;
            shard.dendrite_targets[cell] = (axon_idx as u32) << 8;
            shard.dendrite_weights[cell] = weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::{
        axon_growth::{compute_layer_ranges, grow_axons},
        neuron_placement::place_neurons,
        seed::{seed_from_str, DEFAULT_MASTER_SEED},
    };
    use crate::parser::{anatomy, blueprints, simulation};

    const SIM_SMALL: &str = r#"
[world]
width_um = 500
depth_um = 500
height_um = 1000
[simulation]
tick_duration_us = 100
total_ticks = 1000
master_seed = "GENESIS"
global_density = 0.10
voxel_size_um = 25
signal_speed_um_tick = 50
sync_batch_ticks = 100
"#;
    const ANATOMY_SMALL: &str = r#"
[[layer]]
name = "L4"
height_pct = 0.50
population_pct = 0.60
[layer.composition]
"Vertical_Excitatory"   = 0.80
"Horizontal_Inhibitory" = 0.20

[[layer]]
name = "L2/3"
height_pct = 0.50
population_pct = 0.40
[layer.composition]
"Vertical_Excitatory"   = 0.85
"Horizontal_Inhibitory" = 0.15
"#;
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

[[neuron_type]]
name = "Horizontal_Inhibitory"
threshold = 40000
rest_potential = 10000
leak_rate = 1500
refractory_period = 10
synapse_refractory_period = 5
conduction_velocity = 100
signal_propagation_length = 5
axon_growth_step = 10
homeostasis_penalty = 3000
homeostasis_decay = 15
slot_decay_ltm = 140
slot_decay_wm = 80
sprouting_weight_distance = 0.6
sprouting_weight_power = 0.3
sprouting_weight_explore = 0.1
"#;

    fn build_small_shard() -> (ShardStateSoA, Vec<PlacedNeuron>, Vec<GrownAxon>) {
        let sim = simulation::parse(SIM_SMALL).unwrap();
        let an = anatomy::parse(ANATOMY_SMALL).unwrap();
        let bp = blueprints::parse(BP).unwrap();
        let type_names: Vec<String> = bp.neuron_type.iter().map(|n| n.name.clone()).collect();
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let neurons = place_neurons(&sim, &an, &type_names, master);
        let ranges = compute_layer_ranges(&an, &sim);
        let axons = grow_axons(&neurons, &ranges, &bp.neuron_type, &sim, master);
        let mut shard = ShardStateSoA::new_blank(neurons.len(), axons.len(), 10_000);
        connect_dendrites(&mut shard, &neurons, &axons, &bp.neuron_type, master);
        (shard, neurons, axons)
    }

    #[test]
    fn some_dendrites_connected() {
        let (shard, ..) = build_small_shard();
        let connected = shard.dendrite_targets.iter().filter(|&&t| t != 0).count();
        assert!(connected > 0, "Expected some dendrites connected, got 0");
    }

    #[test]
    fn no_self_connections() {
        let (shard, neurons, axons) = build_small_shard();
        let pn = shard.padded_n;
        for soma_id in 0..neurons.len() {
            for slot in 0..MAX_DENDRITE_SLOTS {
                let target = shard.dendrite_targets[slot * pn + soma_id];
                if target != 0 {
                    let axon_idx = (target >> 8) as usize;
                    assert_ne!(
                        axons[axon_idx].soma_idx, soma_id,
                        "soma {} connected to its own axon!",
                        soma_id
                    );
                }
            }
        }
    }

    #[test]
    fn weights_are_nonzero_for_connected() {
        let (shard, ..) = build_small_shard();
        let _pn = shard.padded_n;
        for i in 0..shard.dendrite_targets.len() {
            if shard.dendrite_targets[i] != 0 {
                assert_ne!(
                    shard.dendrite_weights[i], 0,
                    "connected synapse at index {} has zero weight",
                    i
                );
            }
        }
    }

    #[test]
    fn grid_build_covers_all_axons() {
        let sim = simulation::parse(SIM_SMALL).unwrap();
        let an = anatomy::parse(ANATOMY_SMALL).unwrap();
        let bp = blueprints::parse(BP).unwrap();
        let type_names: Vec<String> = bp.neuron_type.iter().map(|n| n.name.clone()).collect();
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let neurons = place_neurons(&sim, &an, &type_names, master);
        let ranges = compute_layer_ranges(&an, &sim);
        let axons = grow_axons(&neurons, &ranges, &bp.neuron_type, &sim, master);
        let grid = build_axon_grid(&axons);
        let total_in_grid: usize = grid.values().map(|v| v.len()).sum();
        assert_eq!(
            total_in_grid,
            axons.len(),
            "Every axon must appear in grid exactly once"
        );
    }
}

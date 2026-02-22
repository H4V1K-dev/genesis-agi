use crate::bake::seed::{entity_seed, random_f32, shuffle_indices};
use crate::parser::{anatomy::Anatomy, simulation::SimulationConfig};
use genesis_core::types::PackedPosition;

/// Размещённый нейрон в 3D-пространстве.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PlacedNeuron {
    /// Packed voxel coordinate: [Type(4b)|Z(8b)|Y(10b)|X(10b)]
    pub position: PackedPosition,
    /// Индекс нейронного типа из blueprints.neuron_type[]
    pub type_idx: usize,
    /// Имя слоя в котором размещён нейрон (для отладки)
    pub layer_name: String,
}

impl PlacedNeuron {
    /// Распаковать X координату (биты 9:0)
    pub fn x(&self) -> u32 {
        self.position & 0x3FF
    }
    /// Распаковать Y координату (биты 19:10)
    pub fn y(&self) -> u32 {
        (self.position >> 10) & 0x3FF
    }
    /// Распаковать Z координату (биты 27:20)
    pub fn z(&self) -> u32 {
        (self.position >> 20) & 0xFF
    }
    #[allow(dead_code)]
    /// Распаковать type_mask (биты 31:28)
    pub fn type_mask(&self) -> u32 {
        self.position >> 28
    }
}

/// Упаковывает координаты в PackedPosition.
/// Layout: [type(4b) | z(8b) | y(10b) | x(10b)]
pub fn pack_position(x: u32, y: u32, z: u32, type_mask: u32) -> PackedPosition {
    debug_assert!(x < 1024, "X={} exceeds 10-bit range", x);
    debug_assert!(y < 1024, "Y={} exceeds 10-bit range", y);
    debug_assert!(z < 256, "Z={} exceeds 8-bit range", z);
    debug_assert!(
        type_mask < 16,
        "type_mask={} exceeds 4-bit range",
        type_mask
    );
    (type_mask << 28) | (z << 20) | (y << 10) | x
}

/// Размещает все нейроны зоны в 3D-пространстве.
///
/// Алгоритм (02_configuration.md §5.5):
/// 1. `Layer_Budget = floor(population_pct × total_budget)`
/// 2. Для каждого типа: `count = floor(quota × Layer_Budget)`
/// 3. Нейроны размещаются равномерно в [Z_start, Z_end], XY — случайно в [0, width)
/// 4. Детерминированный shuffle через wyhash + master_seed
pub fn place_neurons(
    sim: &SimulationConfig,
    anatomy: &Anatomy,
    type_names: &[String], // blueprints.neuron_type[i].name → индекс
    master_seed: u64,
) -> Vec<PlacedNeuron> {
    let total_budget = sim.neuron_budget();
    let voxel_um = sim.simulation.voxel_size_um;

    let world_w_vox = sim.world.width_um / voxel_um;
    let world_d_vox = sim.world.depth_um / voxel_um;
    let world_h_vox = sim.world.height_um / voxel_um;

    let mut all_neurons: Vec<PlacedNeuron> = Vec::with_capacity(total_budget as usize);

    let mut z_cursor_pct = 0.0f32;

    for layer in &anatomy.layer {
        let layer_budget = (total_budget as f32 * layer.population_pct) as u64;
        let z_start = (z_cursor_pct * world_h_vox as f32) as u32;
        let z_end = ((z_cursor_pct + layer.height_pct) * world_h_vox as f32) as u32;
        let z_range = (z_end - z_start).max(1);
        z_cursor_pct += layer.height_pct;

        // Для каждого типа — разместить floor(quota × budget) нейронов
        for (type_name, &quota) in &layer.composition {
            let count = (layer_budget as f32 * quota) as u64;
            let type_idx = type_names.iter().position(|n| n == type_name).unwrap_or(0);
            // type_mask — просто индекс типа (0..3), 4 бита
            let type_mask = (type_idx & 0xF) as u32;

            // Генерируем `count` позиций с детерминированным shuffle
            let shuffle = shuffle_indices(
                count as usize,
                entity_seed(master_seed, all_neurons.len() as u32),
            );

            for (i, &si) in shuffle.iter().enumerate() {
                // Равномерное распределение по Z внутри слоя
                let z = z_start + (si as u32 % z_range);
                // XY — псевдослучайно через seed
                let pos_seed = entity_seed(master_seed, (all_neurons.len() + i) as u32);
                let x = (random_f32(pos_seed) * world_w_vox as f32) as u32;
                let y = (random_f32(pos_seed.wrapping_mul(6364136223846793005))
                    * world_d_vox as f32) as u32;

                let x = x.min(world_w_vox - 1);
                let y = y.min(world_d_vox - 1);
                let z = z.min(255); // Z 8-bit cap

                all_neurons.push(PlacedNeuron {
                    position: pack_position(x, y, z, type_mask),
                    type_idx,
                    layer_name: layer.name.clone(),
                });
            }
        }
    }

    all_neurons
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::seed::{seed_from_str, DEFAULT_MASTER_SEED};
    use crate::parser::{anatomy, simulation};

    const SIM: &str = r#"
[world]
width_um = 3500
depth_um = 3500
height_um = 10250
[simulation]
tick_duration_us = 100
total_ticks = 10000
master_seed = "GENESIS"
global_density = 0.04
voxel_size_um = 25
signal_speed_um_tick = 50
sync_batch_ticks = 1000
"#;

    const ANATOMY: &str = r#"
[[layer]]
name = "L4"
height_pct = 0.40
population_pct = 0.55
[layer.composition]
"Vertical_Excitatory"   = 0.80
"Horizontal_Inhibitory" = 0.20

[[layer]]
name = "L2/3"
height_pct = 0.60
population_pct = 0.45
[layer.composition]
"Vertical_Excitatory"   = 0.85
"Horizontal_Inhibitory" = 0.15
"#;

    fn make_sim_and_anatomy() -> (SimulationConfig, Anatomy) {
        (
            simulation::parse(SIM).unwrap(),
            anatomy::parse(ANATOMY).unwrap(),
        )
    }

    #[test]
    fn placement_count_matches_budget() {
        let (sim, an) = make_sim_and_anatomy();
        let type_names = vec![
            "Vertical_Excitatory".to_string(),
            "Horizontal_Inhibitory".to_string(),
        ];
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let neurons = place_neurons(&sim, &an, &type_names, master);
        let budget = sim.neuron_budget() as usize;
        // Допуск: floor() на каждом слое может дать -1..0 нейронов
        assert!(
            (neurons.len() as i64 - budget as i64).abs() < 10,
            "placed={} expected≈{}",
            neurons.len(),
            budget
        );
    }

    #[test]
    fn positions_in_world_bounds() {
        let (sim, an) = make_sim_and_anatomy();
        let type_names = vec![
            "Vertical_Excitatory".to_string(),
            "Horizontal_Inhibitory".to_string(),
        ];
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let neurons = place_neurons(&sim, &an, &type_names, master);
        let voxel = sim.simulation.voxel_size_um;
        let max_x = (sim.world.width_um / voxel) as u32;
        let max_y = (sim.world.depth_um / voxel) as u32;
        // Check first 1000 to stay fast
        for n in neurons.iter().take(1000) {
            assert!(n.x() < max_x, "X={} out of bounds (max {})", n.x(), max_x);
            assert!(n.y() < max_y, "Y={} out of bounds (max {})", n.y(), max_y);
        }
    }

    #[test]
    fn placement_is_deterministic() {
        let (sim, an) = make_sim_and_anatomy();
        let type_names = vec![
            "Vertical_Excitatory".to_string(),
            "Horizontal_Inhibitory".to_string(),
        ];
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let a = place_neurons(&sim, &an, &type_names, master);
        let b = place_neurons(&sim, &an, &type_names, master);
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.position == y.position),
            "placement must be bit-identical for same seed"
        );
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let pos = pack_position(512, 256, 128, 3);
        let n = PlacedNeuron {
            position: pos,
            type_idx: 3,
            layer_name: "L4".into(),
        };
        assert_eq!(n.x(), 512);
        assert_eq!(n.y(), 256);
        assert_eq!(n.z(), 128);
        assert_eq!(n.type_mask(), 3);
    }
}

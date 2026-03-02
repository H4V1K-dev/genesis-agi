use genesis_core::config::{SimulationConfig, BlueprintsConfig, IoConfig, WorldConfig, SimulationParams};
use genesis_core::config::io::OutputMap;
use genesis_core::constants::{GXO_MAGIC};
use crate::bake::neuron_placement::PlacedNeuron;
use crate::bake::output_map::{bake_outputs_to_memory};
use genesis_core::coords::pack_position;

use std::collections::HashMap;

fn fake_sim() -> SimulationConfig {
    SimulationConfig {
        world: WorldConfig {
            width_um: 1000,
            depth_um: 1000,
            height_um: 1000,
        },
        simulation: SimulationParams {
            tick_duration_us: 1000,
            total_ticks: 1000,
            master_seed: "test".to_string(),
            global_density: 1.0,
            voxel_size_um: 1, // 1000x1000x1000 voxels
            signal_speed_um_tick: 500,
            sync_batch_ticks: 1000,
            axon_growth_max_steps: 2000,
            segment_length_voxels: 10,
            night_interval_ticks: 10000,
        },
    }
}

fn fake_blueprints() -> BlueprintsConfig {
    genesis_core::config::BlueprintsConfig {
        neuron_types: vec![
            genesis_core::config::NeuronType {
                name: "TypeA".to_string(),
                ..Default::default()
            },
            genesis_core::config::NeuronType {
                name: "TypeB".to_string(),
                ..Default::default()
            },
        ],
    }
}

#[test]
fn test_bake_output_maps_empty() {
    let io = IoConfig::default();
    let sim = fake_sim();
    let blueprints = fake_blueprints();
    let placed = vec![];

    let packed: Vec<u32> = placed.iter().map(|n: &PlacedNeuron| n.position).collect();
    let name_map = HashMap::new();
    let result = bake_outputs_to_memory(&io, sim.world.width_um as f32, sim.world.depth_um as f32, &packed, &name_map);
    assert!(result.gxo_binary.is_empty());
    assert_eq!(result.num_mapped_somas, 0);
}

#[test]
fn test_bake_output_maps_basic_assignment() {
    let mut io = IoConfig::default();
    io.readout_batch_ticks = Some(50);
    io.outputs.push(OutputMap {
        name: "test_map".to_string(),
        source_zone: "V1".to_string(),
        target_type: "ALL".to_string(),
        width: 2,
        height: 2,
        stride: 1,
    }); // 2x2 map, 4 pixels over a 1000x1000 world => each pixel is 500x500

    let sim = fake_sim();
    let blueprints = fake_blueprints();

    // Place 4 neurons, one in each quadrant
    let placed = vec![
        PlacedNeuron { position: pack_position(250, 250, 0, 0), layer_name: String::new(), type_idx: 0 },
        PlacedNeuron { position: pack_position(750, 250, 0, 0), layer_name: String::new(), type_idx: 0 },
        PlacedNeuron { position: pack_position(250, 750, 0, 0), layer_name: String::new(), type_idx: 0 },
        PlacedNeuron { position: pack_position(750, 750, 0, 0), layer_name: String::new(), type_idx: 0 },
    ];

    let packed: Vec<u32> = placed.iter().map(|n: &PlacedNeuron| n.position).collect();
    let name_map = HashMap::new();
    let result = bake_outputs_to_memory(&io, sim.world.width_um as f32, sim.world.depth_um as f32, &packed, &name_map);
    
    assert!(!result.gxo_binary.is_empty());
    assert_eq!(result.num_mapped_somas, 4);

    // Verify binary format roughly
    // We can't use genesis_runtime here because genesis-baker doesn't depend on it.
    // Let's just check the binary length and total_somas bytes manually
    assert!(result.gxo_binary.len() > 20); // Header + maps
    
    // GXO_MAGIC is 0x47584F30
    assert_eq!(&result.gxo_binary[0..4], &GXO_MAGIC.to_le_bytes());
}

#[test]
fn test_bake_output_maps_type_filtering() {
    let mut io = IoConfig::default();
    io.outputs.push(OutputMap {
        name: "only_type_b".to_string(),
        source_zone: "V1".to_string(),
        target_type: "TypeB".to_string(),
        width: 1,
        height: 1,
        stride: 1,
    });

    let sim = fake_sim();
    let blueprints = fake_blueprints();

    let placed = vec![
        PlacedNeuron { position: pack_position(500, 500, 0, 0), layer_name: String::new(), type_idx: 0 }, // TypeA
        PlacedNeuron { position: pack_position(500, 500, 0, 1), layer_name: String::new(), type_idx: 1 }, // TypeB
        PlacedNeuron { position: pack_position(500, 500, 0, 0), layer_name: String::new(), type_idx: 0 }, // TypeA
    ];

    let packed: Vec<u32> = placed.iter().map(|n: &PlacedNeuron| n.position).collect();
    let mut name_map = HashMap::new();
    name_map.insert("TypeA".to_string(), 0);
    name_map.insert("TypeB".to_string(), 1);

    let result = bake_outputs_to_memory(&io, sim.world.width_um as f32, sim.world.depth_um as f32, &packed, &name_map);
    assert_eq!(result.num_mapped_somas, 1);
    
    // Check that there is exactly 1 soma ID written at the end of the binary
    // Header size + map descriptor size + pixel indices size + soma ID
    // 16 bytes header + 44 bytes map + 4 + 2 bytes pixel index + 4 bytes soma id = 70 bytes max roughly
    let mut soma_id_bytes = [0u8; 4];
    let len = result.gxo_binary.len();
    soma_id_bytes.copy_from_slice(&result.gxo_binary[len-4..]);
    let parsed_id = u32::from_le_bytes(soma_id_bytes);
    
    assert_eq!(parsed_id, 1, "Only the second neuron (TypeB) should be selected");
}

use super::*;
use crate::parser::simulation::{SimulationConfig, SimulationParams, WorldConfig};
use genesis_core::config::blueprints::{GenesisConstantMemory, VariantParameters};

fn make_sim_config(speed: u32, len_voxels: u32) -> SimulationConfig {
    SimulationConfig {
        world: WorldConfig {
            width_um: 50,
            depth_um: 50,
            height_um: 50,
        },
        simulation: SimulationParams {
            voxel_size_um: 25,
            segment_length_voxels: len_voxels, // -> v_seg = speed / (25 * len_voxels)
            axon_growth_max_steps: 100,
            tick_duration_us: 1000,
            total_ticks: 100_000,
            master_seed: "0".to_string(),
            global_density: 1.0,
            signal_speed_um_tick: speed,
            sync_batch_ticks: 10,
            night_interval_ticks: 1000,
        },
    }
}

fn make_const_mem_with_prop(prop_len: u8, refr: u8) -> GenesisConstantMemory {
    let mut variants = [VariantParameters::default(); 16];
    variants[0] = VariantParameters {
        signal_propagation_length: prop_len as u16,
        refractory_period: refr,
        ..VariantParameters::default()
    };
    GenesisConstantMemory { variants }
}

#[test]
fn test_validator_rejects_prop_lt_v_seg() {
    // speed=100, voxel=25, voxels_per_seg=1 -> speed=100, seg_um=25 -> v_seg=4
    let sim = make_sim_config(100, 1);
    // prop_len = 3 < v_seg (4) -> Error!
    let const_mem = make_const_mem_with_prop(3, 10);

    let res = check_propagation_covers_v_seg(&sim, &const_mem);
    assert!(res.is_err(), "Validator MUST reject prop_len < v_seg");
}

#[test]
fn test_validator_rejects_refr_gt_prop() {
    // prop = 5, refr = 10 -> Error!
    let const_mem = make_const_mem_with_prop(5, 10);

    let res = check_single_spike_in_flight(&const_mem);
    assert!(res.is_err(), "Validator MUST reject refractory_period > signal_propagation_length");
}

#[test]
fn test_validator_accepts_refr_le_prop() {
    // prop = 10, refr = 5 -> Ok
    let const_mem = make_const_mem_with_prop(10, 5);

    let res = check_single_spike_in_flight(&const_mem);
    assert!(res.is_ok(), "Validator should accept refractory_period <= signal_propagation_length");
}

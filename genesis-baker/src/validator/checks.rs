use crate::parser::{anatomy::Anatomy, blueprints::Blueprints, simulation::SimulationConfig};
use anyhow::bail;

/// Запускает все проверки конфигурации.
/// Возвращает Ok(()) или первую критическую ошибку.
pub fn validate_all(
    sim: &SimulationConfig,
    blueprints: &Blueprints,
    anatomy: &Anatomy,
) -> anyhow::Result<()> {
    check_v_seg_divisible(sim)?;
    check_layer_height_sum(anatomy)?;
    check_layer_population_sum(anatomy)?;
    check_sprouting_weights(blueprints)?;
    check_composition_sums(anatomy)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// §2.3 — v_seg делимость (01_foundations.md §1.6)
// ---------------------------------------------------------------------------

/// `signal_speed_um_tick` обязан делиться на `segment_length_um` без остатка.
/// Иначе v_seg (скорость в сегментах/тик) не будет целым числом → недетерминизм.
pub fn check_v_seg_divisible(sim: &SimulationConfig) -> anyhow::Result<()> {
    let speed = sim.simulation.signal_speed_um_tick as u32;
    let voxel = sim.simulation.voxel_size_um;

    // segment_length_um = voxel_size_um (1 сегмент = 1 воксель по умолчанию)
    if !speed.is_multiple_of(voxel) {
        bail!(
            "CRITICAL: signal_speed_um_tick ({}) must be divisible by voxel_size_um ({}).\n\
             v_seg = signal_speed / voxel_size = {}/{} — не целое число.\n\
             Нарушает Integer Physics детерминизм (01_foundations.md §1.6).",
            speed,
            voxel,
            speed,
            voxel
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// anatomy.toml — суммы height_pct и population_pct
// ---------------------------------------------------------------------------

/// Сумма `height_pct` всех слоёв должна быть ≈ 1.0.
/// Биологический инвариант: слои покрывают всю высоту зоны.
pub fn check_layer_height_sum(anatomy: &Anatomy) -> anyhow::Result<()> {
    let sum: f32 = anatomy.layer.iter().map(|l| l.height_pct).sum();
    if (sum - 1.0).abs() > 0.01 {
        bail!(
            "anatomy.toml: Σ(layer.height_pct) = {:.4} ≠ 1.0 (±0.01).\n\
             Слои обязаны покрывать всю высоту зоны без перекрытий и пробелов.",
            sum
        );
    }
    Ok(())
}

/// Сумма `population_pct` всех слоёв должна быть ≈ 1.0.
pub fn check_layer_population_sum(anatomy: &Anatomy) -> anyhow::Result<()> {
    let sum: f32 = anatomy.layer.iter().map(|l| l.population_pct).sum();
    if (sum - 1.0).abs() > 0.01 {
        bail!(
            "anatomy.toml: Σ(layer.population_pct) = {:.4} ≠ 1.0 (±0.01).\n\
             Бюджет нейронов должен быть распределён полностью.",
            sum
        );
    }
    Ok(())
}

/// Сумма весов composition каждого слоя должна быть ≈ 1.0.
pub fn check_composition_sums(anatomy: &Anatomy) -> anyhow::Result<()> {
    for layer in &anatomy.layer {
        let sum: f32 = layer.composition.values().sum();
        if (sum - 1.0).abs() > 0.01 {
            bail!(
                "anatomy.toml: Layer '{}' Σ(composition) = {:.4} ≠ 1.0 (±0.01).\n\
                 Квоты типов нейронов в слое обязаны суммироваться в 1.0.",
                layer.name,
                sum
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// blueprints.toml — sprouting weights
// ---------------------------------------------------------------------------

/// Сумма sprouting weights каждого типа должна быть ≈ 1.0.
/// (04_connectivity.md §1.6.1: weight_distance + weight_power + weight_explore)
pub fn check_sprouting_weights(blueprints: &Blueprints) -> anyhow::Result<()> {
    for nt in &blueprints.neuron_type {
        let sum = nt.sprouting_weight_sum();
        if (sum - 1.0).abs() > 0.02 {
            bail!(
                "blueprints.toml: NeuronType '{}' sprouting weights sum = {:.4} ≠ 1.0 (±0.02).\n\
                 (weight_distance + weight_power + weight_explore должны суммироваться в 1.0)",
                nt.name,
                sum
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{anatomy, blueprints, simulation};

    const SIM_OK: &str = r#"
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

    const ANATOMY_OK: &str = r#"
[[layer]]
name = "L4"
height_pct = 0.60
population_pct = 0.60
[layer.composition]
"Vertical_Excitatory" = 0.80
"Horizontal_Inhibitory" = 0.20

[[layer]]
name = "L2/3"
height_pct = 0.40
population_pct = 0.40
[layer.composition]
"Vertical_Excitatory" = 0.85
"Horizontal_Inhibitory" = 0.15
"#;

    const BP_OK: &str = r#"
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
sprouting_weight_power   = 0.4
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
sprouting_weight_power   = 0.3
sprouting_weight_explore = 0.1
"#;

    #[test]
    fn valid_config_passes() {
        let sim = simulation::parse(SIM_OK).unwrap();
        let bp = blueprints::parse(BP_OK).unwrap();
        let an = anatomy::parse(ANATOMY_OK).unwrap();
        validate_all(&sim, &bp, &an).expect("valid config must pass");
    }

    #[test]
    fn v_seg_not_divisible_fails() {
        // speed=50, voxel=30 → 50%30 ≠ 0
        let src = SIM_OK.replace("voxel_size_um = 25", "voxel_size_um = 30");
        let sim = simulation::parse(&src).unwrap();
        let err = check_v_seg_divisible(&sim).unwrap_err();
        assert!(err.to_string().contains("CRITICAL"));
    }

    #[test]
    fn height_pct_sum_wrong_fails() {
        let bad = ANATOMY_OK.replace("height_pct = 0.40", "height_pct = 0.50");
        let an = anatomy::parse(&bad).unwrap();
        assert!(check_layer_height_sum(&an).is_err());
    }

    #[test]
    fn sprouting_weights_bad_fails() {
        let bad = BP_OK.replace(
            "sprouting_weight_explore = 0.1\n\n[[neuron_type]]",
            "sprouting_weight_explore = 0.5\n\n[[neuron_type]]",
        );
        let bp = blueprints::parse(&bad).unwrap();
        assert!(check_sprouting_weights(&bp).is_err());
    }
}

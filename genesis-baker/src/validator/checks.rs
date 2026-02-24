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



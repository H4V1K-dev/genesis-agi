use serde::Deserialize;
use std::collections::HashMap;

/// Полный `anatomy.toml` — список слоёв зоны.
#[derive(Debug, Deserialize)]
pub struct Anatomy {
    pub layer: Vec<Layer>,
}

/// Один [[layer]] блок из anatomy.toml.
#[derive(Debug, Deserialize)]
pub struct Layer {
    /// Имя слоя, например "L1", "L4", "Nuclear".
    pub name: String,
    /// Высота слоя как доля от world.height_um (0.0..1.0).
    pub height_pct: f32,
    /// Доля от общего нейронного бюджета зоны (0.0..1.0).
    pub population_pct: f32,
    /// Жёсткие квоты: {type_name → fraction}. Сумма должна быть = 1.0.
    pub composition: HashMap<String, f32>,
}

impl Anatomy {
    /// Рассчитывает абсолютное число нейронов каждого типа в каждом слое.
    /// Возвращает: Vec<(layer_name, type_name, count)>
    #[allow(dead_code)]
    pub fn neuron_counts(&self, total_budget: u64) -> Vec<(String, String, u64)> {
        let mut result = Vec::new();
        for layer in &self.layer {
            let layer_budget = (total_budget as f64 * layer.population_pct as f64) as u64;
            for (type_name, &quota) in &layer.composition {
                let count = (layer_budget as f64 * quota as f64) as u64;
                result.push((layer.name.clone(), type_name.clone(), count));
            }
        }
        result
    }
}

/// Парсит `anatomy.toml` из строки.
pub fn parse(src: &str) -> anyhow::Result<Anatomy> {
    let anatomy: Anatomy = toml::from_str(src)?;
    Ok(anatomy)
}


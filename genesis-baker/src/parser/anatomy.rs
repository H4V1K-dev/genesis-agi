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

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
[[layer]]
name = "L4"
height_pct = 0.17
population_pct = 0.35

[layer.composition]
"Vertical_Excitatory"  = 0.80
"Horizontal_Inhibitory" = 0.20

[[layer]]
name = "L2/3"
height_pct = 0.25
population_pct = 0.29

[layer.composition]
"Vertical_Excitatory"  = 0.85
"Horizontal_Inhibitory" = 0.15
"#;

    #[test]
    fn parse_anatomy_example() {
        let a = parse(EXAMPLE).expect("parse failed");
        assert_eq!(a.layer.len(), 2);
        let l4 = &a.layer[0];
        assert_eq!(l4.name, "L4");
        assert!((l4.height_pct - 0.17).abs() < 1e-5);
        assert!((l4.population_pct - 0.35).abs() < 1e-5);
        assert_eq!(l4.composition.len(), 2);
    }

    #[test]
    fn neuron_counts_hard_quotas() {
        let a = parse(EXAMPLE).expect("parse failed");
        // budget=321440 (из simulation test)
        let counts = a.neuron_counts(321_440);
        // L4: 321440 * 0.35 = 112504 → VE: 90003, HI: 22500
        let l4_ve = counts
            .iter()
            .find(|(l, t, _)| l == "L4" && t == "Vertical_Excitatory")
            .map(|(_, _, c)| *c)
            .unwrap_or(0);
        assert!(
            l4_ve > 85_000 && l4_ve < 100_000,
            "L4 VE count out of range: {}",
            l4_ve
        );
    }
}

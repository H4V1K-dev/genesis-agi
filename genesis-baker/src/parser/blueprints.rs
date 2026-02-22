use serde::Deserialize;

/// Полный `blueprints.toml` — список типов нейронов.
/// Парсится как `table.neuron_type = [...]`.
#[derive(Debug, Deserialize)]
pub struct Blueprints {
    pub neuron_type: Vec<NeuronType>,
}

/// Один [[neuron_type]] блок из blueprints.toml.
/// Все числовые параметры соответствуют Integer Physics (02_configuration.md §6).
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct NeuronType {
    /// Уникальное имя типа. Используется как ключ в anatomy.toml [[layer.composition]].
    pub name: String,

    // --- Параметры Мембраны (i32, microVolts) ---
    /// Порог срабатывания спайка.
    pub threshold: i32,
    /// Потенциал покоя (начальный voltage при старте и после спайка).
    pub rest_potential: i32,
    /// Скорость утечки — вычитается из voltage каждый тик.
    pub leak_rate: i32,

    // --- Тайминги (u8, Ticks) ---
    /// Абсолютная рефрактерность сомы (тики).
    pub refractory_period: u8,
    /// Рефрактерность входного дендритного порта (тики).
    pub synapse_refractory_period: u8,

    // --- Физика Сигнала (u16) ---
    /// Скорость проводимости (дискретное смещение head за тик).
    pub conduction_velocity: u16,
    /// Длина Active Tail в сегментах (PROPAGATION_LENGTH).
    pub signal_propagation_length: u16,
    /// Шаг роста аксона в вокселях (Cone Tracing step).
    pub axon_growth_step: u16,

    // --- Гомеостаз (Adaptive Threshold) ---
    /// Штраф к threshold_offset после спайка.
    pub homeostasis_penalty: i32,
    /// Декремент threshold_offset каждый тик.
    pub homeostasis_decay: u16,

    // --- Slot Decay (Fixed-point: 128 = 1.0×) ---
    /// LTM слоты 0-79: множитель удержания веса.
    pub slot_decay_ltm: u8,
    /// WM слоты 80-127: множитель распада веса.
    pub slot_decay_wm: u8,

    // --- Sprouting Score Weights (f32, sum должна быть ≈ 1.0) ---
    /// Вес близости при выборе аксона-кандидата.
    pub sprouting_weight_distance: f32,
    /// Вес soma_power_index целевой сомы.
    pub sprouting_weight_power: f32,
    /// Вес случайного шума (защита от зацикливания).
    pub sprouting_weight_explore: f32,
}

impl NeuronType {
    /// Суммарный вес sprouting score (должна быть ≈ 1.0).
    pub fn sprouting_weight_sum(&self) -> f32 {
        self.sprouting_weight_distance + self.sprouting_weight_power + self.sprouting_weight_explore
    }
}

/// Парсит `blueprints.toml` из строки.
pub fn parse(src: &str) -> anyhow::Result<Blueprints> {
    let bp: Blueprints = toml::from_str(src)?;
    Ok(bp)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
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
    fn parse_blueprints_example() {
        let bp = parse(EXAMPLE).expect("parse failed");
        assert_eq!(bp.neuron_type.len(), 2);

        let ve = &bp.neuron_type[0];
        assert_eq!(ve.name, "Vertical_Excitatory");
        assert_eq!(ve.threshold, 42000);
        assert_eq!(ve.rest_potential, 10000);
        assert_eq!(ve.refractory_period, 15);
        assert_eq!(ve.slot_decay_ltm, 160);
        assert_eq!(ve.slot_decay_wm, 96);
        // sprouting weights sum ≈ 1.0
        assert!((ve.sprouting_weight_sum() - 1.0).abs() < 1e-5);

        let hi = &bp.neuron_type[1];
        assert_eq!(hi.name, "Horizontal_Inhibitory");
        assert_eq!(hi.conduction_velocity, 100);
        assert_eq!(hi.slot_decay_wm, 80);
    }
}

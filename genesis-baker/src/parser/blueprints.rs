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
#[derive(Debug, Deserialize, Clone)]
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

    // --- Рост аксона (Steering) ---
    /// Угол поля зрения (FOV) конуса поиска в градусах.
    pub steering_fov_deg: f32,
    /// Максимальный радиус поиска для конуса (в микрометрах).
    pub steering_radius_um: f32,
    /// Вес сохранения текущего направления роста.
    pub steering_weight_inertia: f32,
    /// Вес притяжения к совместимым нейронам локально.
    pub steering_weight_sensor: f32,
    /// Вес случайного отклонения.
    pub steering_weight_jitter: f32,

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


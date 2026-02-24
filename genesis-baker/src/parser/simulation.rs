use serde::Deserialize;

/// Полный `simulation.toml`.
#[derive(Debug, Deserialize)]
pub struct SimulationConfig {
    pub world: World,
    pub simulation: Simulation,
}

/// [world] — Физические размеры пространства (в микрометрах).
#[derive(Debug, Deserialize)]
pub struct World {
    pub width_um: u32,
    pub depth_um: u32,
    pub height_um: u32,
}

/// [simulation] — Глобальные параметры симуляции.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct Simulation {
    /// Шаг времени в микросекундах. 100 = 0.1 мс.
    pub tick_duration_us: u32,
    /// Количество тиков в симуляции (может быть 0 = бесконечно).
    pub total_ticks: u64,
    /// Глобальный сид (строка, хэшируется в u64 при запуске).
    pub master_seed: String,
    /// Процент вокселей с телами нейронов (0.0..1.0).
    pub global_density: f32,
    /// Размер вокселя (квант пространства в микрометрах).
    pub voxel_size_um: u32,
    /// Скорость распространения сигнала (мкм/тик).
    pub signal_speed_um_tick: u16,
    /// Количество тиков автономного расчета между синхронизациями шардов.
    pub sync_batch_ticks: u32,
    /// Длина одного сегмента аксона в вокселях (глобальная, фиксированная).
    #[serde(default = "default_segment_length")]
    pub segment_length_voxels: u32,
    /// Количество виртуальных аксонов (сетчатка). Опционально.
    pub num_virtual_axons: Option<u32>,
    /// Максимальное количество шагов роста аксона (предохранитель от бесконечных циклов).
    #[serde(default = "default_max_steps")]
    pub axon_growth_max_steps: u32,
}

fn default_segment_length() -> u32 { 5 }
fn default_max_steps() -> u32 { 2000 }

impl SimulationConfig {
    /// Общее число вокселей для заданного размера вокселя (в мкм).
    pub fn total_voxels(&self) -> u64 {
        let v_um = self.simulation.voxel_size_um;
        let w = (self.world.width_um / v_um) as u64;
        let d = (self.world.depth_um / v_um) as u64;
        let h = (self.world.height_um / v_um) as u64;
        w * d * h
    }

    /// Максимальное число нейронов = total_voxels * global_density.
    pub fn neuron_budget(&self) -> u64 {
        (self.total_voxels() as f64 * self.simulation.global_density as f64) as u64
    }
}

/// Парсит `simulation.toml` из строки.
pub fn parse(src: &str) -> anyhow::Result<SimulationConfig> {
    let cfg: SimulationConfig = toml::from_str(src)?;
    Ok(cfg)
}


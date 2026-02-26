use serde::Deserialize;

/// Represents external projection connections coming into this shard (White Matter/Atlas).
#[derive(Debug, Deserialize, Clone, Default)]
pub struct IoConfig {
    #[serde(default)]
    #[serde(rename = "input")]
    pub inputs: Vec<InputMap>,

    #[serde(default)]
    #[serde(rename = "output")]
    pub outputs: Vec<OutputMap>,

    /// Количество тиков в одном батче вывода (по умолчанию равно размеру sync_batch_ticks)
    #[serde(default)]
    pub readout_batch_ticks: Option<u32>,
}


#[derive(Debug, Deserialize, Clone)]
pub struct InputMap {
    /// Имя признака/канала, например "retina_edges"
    pub name: String,
    
    /// Название зоны куда инжектится этот ввод, например "V1"
    pub target_zone: String,
    
    /// Тип нейрона к которому нужно подключаться, например "L4_Stellate".
    /// Используйте "ALL" чтобы не ограничивать выбор типом.
    pub target_type: String,
    
    /// Ширина входной матрицы в пикселях
    pub width: u32,
    
    /// Высота входной матрицы в пикселях
    pub height: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OutputMap {
    /// Имя популяции/канала вывода, например "motor_arm"
    pub name: String,
    
    /// Название зоны из которой читаем спайки, например "M1"
    pub source_zone: String,
    
    /// Фрагмент маски (тип нейронов), которые входят в популяцию тайла.
    /// Используйте "ALL" для сбора всех типов.
    pub target_type: String,
    
    /// Ширина выходной матрицы в тайлах
    pub width: u32,
    
    /// Высота выходной матрицы в тайлах
    pub height: u32,
}

impl IoConfig {
    /// Парсит конфиг из TOML строки.
    pub fn parse(src: &str) -> Result<Self, String> {
        toml::from_str(src).map_err(|e| format!("TOML parse error: {}", e))
    }

    /// Загружает конфиг с диска.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {:?}: {}", path, e))?;
        Self::parse(&content)
    }
}

#[cfg(test)]
#[path = "test_io.rs"]
mod test_io;

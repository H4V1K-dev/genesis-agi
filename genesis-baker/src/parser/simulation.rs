//! Парсер симуляции (делегируется к `genesis_core::config`).

pub use genesis_core::config::{SimulationConfig, SimulationParams, WorldConfig};

/// Парсит `simulation.toml` из строки, конвертируя `String` ошибку в `anyhow::Result`.
pub fn parse(src: &str) -> anyhow::Result<SimulationConfig> {
    SimulationConfig::parse(src).map_err(|e| anyhow::anyhow!(e))
}


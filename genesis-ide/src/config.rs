use bevy::prelude::*;
use serde::Deserialize;

#[derive(Resource, Deserialize, Clone, Debug)]
pub struct IdeConfig {
    pub target_ip: String,
    pub geom_port: u16,
    pub telemetry_port: u16,
}

impl Default for IdeConfig {
    fn default() -> Self {
        Self {
            target_ip: "127.0.0.1".to_string(),
            geom_port: 9002,       // Fallback (default Genesis node port)
            telemetry_port: 9003,  // Fallback (default Genesis node port)
        }
    }
}

/// Инициализация конфигурации до старта сетевых систем
/// Запускается в PreStartup расписании, гарантируя доступность ресурса для других систем
pub fn parse_cli_config(mut commands: Commands) {
    let args: Vec<String> = std::env::args().collect();
    let mut config = IdeConfig::default();

    // Простой O(N) парсер CLI без тяжелых зависимостей типа clap для MVP
    if let Some(pos) = args.iter().position(|a| a == "--geom") {
        if pos + 1 < args.len() {
            config.geom_port = args[pos + 1]
                .parse()
                .expect("Invalid geometry port: expected u16");
        }
    }

    if let Some(pos) = args.iter().position(|a| a == "--telemetry") {
        if pos + 1 < args.len() {
            config.telemetry_port = args[pos + 1]
                .parse()
                .expect("Invalid telemetry port: expected u16");
        }
    }

    if let Some(pos) = args.iter().position(|a| a == "--ip") {
        if pos + 1 < args.len() {
            config.target_ip = args[pos + 1].clone();
        }
    }

    println!(
        "[IDE Config] Initialized: {}:{} (Geom), {}:{} (Telemetry)",
        config.target_ip, config.geom_port, config.target_ip, config.telemetry_port
    );

    commands.insert_resource(config);
}

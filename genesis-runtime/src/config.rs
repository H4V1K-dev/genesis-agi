use serde::Deserialize;
use std::path::Path;
use anyhow::{Context, Result};
use std::fs;

/// Configures exactly *what* piece of the brain this node simulates, 
/// and *where* its neighbors are located.
#[derive(Debug, Deserialize, Clone)]
pub struct InstanceConfig {
    /// Reference to the zone folder name (e.g. "V1")
    pub zone_id: String,
    
    /// Offset of this shard in the global brain space (in voxels)
    pub world_offset: Coordinate,
    
    /// Dimensions of this shard (in voxels)
    pub dimensions: Dimensions,
    
    /// Neighborhood topology. "Self" means loopback (toroidal graph mapping),
    /// otherwise an IP:Port string. Left blank if bounded.
    pub neighbors: Neighbors,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Coordinate {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Dimensions {
    pub w: u32,
    pub d: u32,
    pub h: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Neighbors {
    pub x_plus: Option<String>,
    pub x_minus: Option<String>,
    pub y_plus: Option<String>,
    pub y_minus: Option<String>,
}

/// Parses an Instance Config (e.g. `shard_04.toml`) from disk.
pub fn parse_shard_config(path: &Path) -> Result<InstanceConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read InstanceConfig at {:?}", path))?;

    let config: InstanceConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse TOML InstanceConfig from {:?}", path))?;

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_shard_config_example() {
        let toml_str = r#"
zone_id = "V1"

[world_offset]
x = 1000
y = 0
z = 0

[dimensions]
w = 500
d = 500
h = 2000

[neighbors]
x_plus = "192.168.1.10:8000"
x_minus = "Self"
y_plus = "192.168.1.11:8000"
# y_minus is bounded (None)
        "#;

        let config: InstanceConfig = toml::from_str(toml_str).expect("Failed to parse string");
        assert_eq!(config.zone_id, "V1");
        assert_eq!(config.world_offset.x, 1000);
        assert_eq!(config.world_offset.z, 0);
        assert_eq!(config.dimensions.h, 2000);
        assert_eq!(config.neighbors.x_minus.as_deref(), Some("Self"));
        assert_eq!(config.neighbors.y_minus, None);
    }
}

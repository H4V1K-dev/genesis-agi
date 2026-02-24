use serde::Deserialize;

/// Represents external projection connections coming into this shard (White Matter/Atlas).
#[derive(Debug, Deserialize)]
pub struct IoConfig {
    pub input: Vec<InputChannel>,
}

#[derive(Debug, Deserialize)]
pub struct InputChannel {
    /// Friendly name describing the origin of this channel (e.g., "From_LGN")
    pub name: String,
    
    /// The target layer inside this zone where these external axons will spread
    pub target_layer: String,
    
    /// Hard quota of axons arriving from this external source
    pub axon_count: u32,
    
    /// The phenotype mask to assign to these incoming segments
    /// Affects downstream plasticity (GSOP LUT applied by the dendrites)
    pub type_mask: u8,
}

/// Parse `io.toml` from a string
pub fn parse(src: &str) -> anyhow::Result<IoConfig> {
    let io: IoConfig = toml::from_str(src)?;
    Ok(io)
}


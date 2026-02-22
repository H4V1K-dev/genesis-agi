use anyhow::{Context, Result};
use clap::Parser;
use genesis_runtime::config::{parse_shard_config, InstanceConfig};
use genesis_runtime::memory::VramState;
use genesis_runtime::orchestrator::night_phase::NightPhase;
use genesis_runtime::Runtime;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "genesis-node",
    about = "Distributed Genesis Brain Node Daemon",
    version
)]
struct Cli {
    /// Path to the shard configuration (e.g. shard_04.toml)
    #[arg(short, long)]
    config: PathBuf,

    /// Directory containing baked binary blocks (.state, .axons)
    #[arg(short, long, default_value = "baked/")]
    baked_dir: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse CLI
    let cli = Cli::parse();
    println!("[Node] Starting Genesis Distributed Daemon...");

    // 2. Load Instance Config
    let config = parse_shard_config(&cli.config)
        .with_context(|| format!("Failed to load config: {:?}", cli.config))?;
    
    println!("[Node] Target Zone: {}", config.zone_id);
    println!(
        "[Node] World Offset: [{}, {}, {}]",
        config.world_offset.x, config.world_offset.y, config.world_offset.z
    );

    // 3. Load Baked Geometry (Zero-Copy to VRAM)
    // We expect baker to produce shard_{zone_id}.state or similar. 
    // Here we'll just look for a generic shard.state / shard.axons for the MVP.
    let state_path = cli.baked_dir.join("shard.state");
    let axons_path = cli.baked_dir.join("shard.axons");

    println!("[Node] Loading VRAM payload from {:?}...", cli.baked_dir);
    let state_bytes = std::fs::read(&state_path).context("Missing shard.state")?;
    let axons_bytes = std::fs::read(&axons_path).context("Missing shard.axons")?;

    let vram = VramState::load_shard(&state_bytes, &axons_bytes)
        .context("Failed to push shard data to GPU VRAM")?;

    println!("[Node] VRAM Load Complete. {} neurons, {} axons", vram.padded_n, vram.total_axons);

    // 4. Initialize Network (Stubbed for now, full integration later)
    // - UDP Router
    // - TCP Geometry Server

    // 5. Initialize Engine Runtime
    // (We pass empty CPU lists for neurons/axons because the Baker hasn't supplied them in this simplified harness)
    let v_seg = 3; // Placeholder inherited from global config
    let mut runtime = Runtime::new(
        vram,
        v_seg,
        Arc::new(vec![]),
        Arc::new(vec![]),
        Arc::new(vec![]),
        0, // master seed
    );

    // 6. Enter the Ephemeral Loop
    let mut current_tick = 0u64;
    let night_interval = 10000; // placeholder
    
    println!("[Node] Engine Online. Starting Ephemeral Loop.");

    loop {
        let loop_start = Instant::now();

        // 6.1 Night Phase Check
        let is_night = NightPhase::check_and_run(&mut runtime, 0, night_interval, current_tick);

        if is_night {
            // Re-sync local states if needed
            println!("[Node] Night Phase concluded at tick {}.", current_tick);
        }

        // 6.2 Day Phase: Tick GPU
        runtime.tick();
        
        // 6.3 Fast Path Network Sync (TODO)

        // Sync GPU pipeline before next loop
        runtime.synchronize();

        current_tick += 1;

        // Throttle to simulated real time (e.g. 100us tick = 10k ticks per second)
        // Here we just limit to 100ms for console readability in our test
        if current_tick % 1000 == 0 {
            println!("[Node] Processed 1000 ticks. Total: {}", current_tick);
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

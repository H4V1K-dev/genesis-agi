use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

mod bake;
mod parser;
mod validator;

#[derive(Parser)]
#[command(
    name = "baker",
    about = "Genesis Baking Tool — TOML configs → binary .state/.axons blobs"
)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile TOML configs into binary blobs ready for GPU loading.
    Compile {
        #[arg(long, default_value = "config/simulation.toml")]
        simulation: PathBuf,
        #[arg(long, default_value = "config/zones/V1/blueprints.toml")]
        blueprints: std::path::PathBuf,
        #[arg(long, default_value = "config/zones/V1/anatomy.toml")]
        anatomy: std::path::PathBuf,
        #[arg(long, default_value = "config/zones/V1/io.toml")]
        io: PathBuf,
        #[arg(short, long, default_value = "baked/")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Compile {
            simulation,
            blueprints,
            anatomy,
            io,
            output,
        } => compile(&simulation, &blueprints, &anatomy, &io, &output),
    }
}

fn compile(sim_path: &Path, bp_path: &Path, an_path: &Path, io_path: &Path, out_dir: &Path) -> Result<()> {
    // --- 1. Parse ---
    println!("[baker] Parsing configs...");
    let sim = parser::simulation::parse(
        &std::fs::read_to_string(sim_path)
            .with_context(|| format!("Cannot read {}", sim_path.display()))?,
    )
    .context("Failed to parse simulation.toml")?;
    let blueprints = parser::blueprints::parse(
        &std::fs::read_to_string(bp_path)
            .with_context(|| format!("Cannot read {}", bp_path.display()))?,
    )
    .context("Failed to parse blueprints.toml")?;
    let anatomy = parser::anatomy::parse(
        &std::fs::read_to_string(an_path)
            .with_context(|| format!("Cannot read {}", an_path.display()))?,
    )
    .context("Failed to parse anatomy.toml")?;
    // IO is technically optional for an isolated shard, but we parse it if it exists.
    let io = if io_path.exists() {
        Some(
            parser::io::parse(
                &std::fs::read_to_string(io_path)
                    .with_context(|| format!("Cannot read {}", io_path.display()))?,
            )
            .context("Failed to parse io.toml")?,
        )
    } else {
        None
    };
    println!(
        "[baker] ✓ Parsed: {} neuron types, {} layers",
        blueprints.neuron_types.len(),
        anatomy.layers.len()
    );

    // --- 2. Validate ---
    println!("[baker] Validating invariants...");
    validator::validate_all(&sim, &blueprints, &anatomy).context("Config validation failed")?;
    println!("[baker] ✓ All invariants passed");

    // --- 3. Master seed ---
    let master_seed_str = &sim.simulation.master_seed;
    let master_seed = bake::seed::seed_from_str(master_seed_str);
    println!("[baker] Master seed: \"{}\"", master_seed_str);

    // --- 4. Place neurons ---
    println!("[baker] Placing neurons...");
    let type_names: Vec<String> = blueprints
        .neuron_types
        .iter()
        .map(|n| n.name.clone())
        .collect();
    let neurons = bake::neuron_placement::place_neurons(&sim, &anatomy, &type_names, master_seed);
    println!("[baker] ✓ Placed {} neurons", neurons.len());

    // --- 5. Cone Tracing: grow axons ---
    println!("[baker] Growing axons (Cone Tracing)...");
    let layer_ranges = bake::axon_growth::compute_layer_ranges(&anatomy, &sim);
    let shard_bounds = bake::axon_growth::ShardBounds::full_world(&sim);
    let (mut axons, mut ghost_packets) = bake::axon_growth::grow_axons(
        &neurons,
        &layer_ranges,
        &blueprints.neuron_types,
        &sim,
        &shard_bounds,
        master_seed,
    );
    let local_axons_count = axons.len();

    // --- 5.2 External Inputs (Virtual Axons) ---
    let mut num_virtual = 0;
    if let Some(io_cfg) = &io {
        let shard_name = out_dir.file_name().and_then(|n| n.to_str()).unwrap_or("shard");
        let zone_name = out_dir.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or(shard_name); // simple heuristic
        // Если структура baked/V1/shard_0, то zone_name=V1. Если baked/V1, то out_dir=V1, zone_name=baked (не оч), но target_zone обычно это имя папки.
        // Более точный хак: если есть io.toml, то zone - это target_zone из него. Но io.toml общий для зоны.
        // Подождите: у нас нет --zone аргумента. Возьмём первый input, или пусть target_zone совпадает с out_dir.
        
        // Для простоты, вытащим zone_name из io.toml, где target_zone совпадает с чем-то, или просто применим все подходящие по эвристике.
        // Запросим для "V1", или пусть grow_input_maps ищет по .gxi? 
        // Если baker вызывается для конкретной зоны, то IDE передает `--output baked/ZONENAME`.
        let zone_name_heuristic = out_dir.file_name().and_then(|n| n.to_str()).unwrap_or("V1");

        println!("[baker] Processing Input Maps for zone '{}'...", zone_name_heuristic);
        
        let mut virtual_result = bake::input_map::grow_input_maps(
            io_cfg,
            zone_name_heuristic,
            &neurons,
            &layer_ranges,
            &blueprints,
            &sim,
            &shard_bounds,
            master_seed,
            axons.len() as u32,
        );

        num_virtual = virtual_result.axons.len();
        axons.append(&mut virtual_result.axons);
        
        if !virtual_result.ghosts.is_empty() {
            println!("[baker] ⚠ {} virtual axon(s) generated ghost packets.", virtual_result.ghosts.len());
            ghost_packets.append(&mut virtual_result.ghosts);
        }

        if !virtual_result.gxi_binary.is_empty() {
            let gxi_path = out_dir.join(format!("{}.gxi", shard_name));
            atomic_write(&gxi_path, &virtual_result.gxi_binary)?;
            println!(
                "[baker] ✓ Written: {}.gxi ({:.1} KB) with {} virtual axons",
                shard_name,
                virtual_result.gxi_binary.len() as f64 / 1024.0,
                num_virtual
            );
        }

        // --- 5.3 External Outputs (Readout Maps) ---
        println!("[baker] Processing Output Maps for zone '{}'...", zone_name_heuristic);
        
        let output_result = bake::output_map::bake_output_maps(
            io_cfg,
            zone_name_heuristic,
            &neurons,
            &blueprints,
            &sim,
        );

        if !output_result.gxo_binary.is_empty() {
            let gxo_path = out_dir.join(format!("{}.gxo", shard_name));
            atomic_write(&gxo_path, &output_result.gxo_binary)?;
            println!(
                "[baker] ✓ Written: {}.gxo ({:.1} KB) with {} mapped somas",
                shard_name,
                output_result.gxo_binary.len() as f64 / 1024.0,
                output_result.num_mapped_somas
            );
        }
    }
    if !ghost_packets.is_empty() {
        println!("[baker] ✓ {} ghost packet(s) detected — injecting into shard B...", ghost_packets.len());
        let (mut ghost_axons, leftover) = bake::axon_growth::inject_ghost_axons(
            &ghost_packets,
            &neurons,
            &blueprints.neuron_types,
            &sim,
            &shard_bounds,
            master_seed,
        );
        if !leftover.is_empty() {
            println!("[baker] ⚠ {} ghost(s) exited shard B boundary (further shards not yet wired)", leftover.len());
        }
        println!("[baker] ✓ Injected {} ghost axons", ghost_axons.len());
        axons.append(&mut ghost_axons);
    }

    // --- 5.5 Atlas Routing (White Matter) ---
    // Временно отключено, будет переработано под новый IoConfig

    println!("[baker] ✓ Total Grown: {} axons ({} local, {} virtual)", 
        axons.len(), local_axons_count, num_virtual);

    // --- 6. Build SoA state ---
    let rest_potential = blueprints
        .neuron_types
        .first()
        .map(|n| n.rest_potential)
        .unwrap_or(10_000);
    let mut shard =
        bake::layout::ShardStateSoA::new_blank(neurons.len(), axons.len(), rest_potential);

    // --- 7. Init axon heads (not zero, not SENTINEL) ---
    let physics = genesis_core::physics::compute_derived_physics(
        sim.simulation.signal_speed_um_tick as u32,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    ).expect("v_seg validation failed. Run baker validation first.");
    let v_seg = physics.v_seg;
    for (i, ax) in axons.iter().enumerate() {
        if i < shard.axon_heads.len() {
            shard.axon_heads[i] = bake::axon_growth::init_axon_head(ax.length_segments, v_seg);
        }
        // soma_to_axon маппинг: сома i → аксон i
        if ax.soma_idx != usize::MAX && ax.soma_idx < shard.soma_to_axon.len() {
            shard.soma_to_axon[ax.soma_idx] = i as u32;
        }
    }
    println!("[baker] ✓ Axon heads initialized (v_seg={})", v_seg);

    // --- 8. Connect dendrites (spatial search + sprouting score) ---
    println!("[baker] Connecting dendrites...");
    bake::dendrite_connect::connect_dendrites(
        &mut shard,
        &neurons,
        &axons,
        &blueprints.neuron_types,
        master_seed,
    );
    let connected = shard.dendrite_targets.iter().filter(|&&t| t != 0).count();
    println!(
        "[baker] ✓ Connected {} synapses ({:.1} avg per neuron)",
        connected,
        connected as f64 / neurons.len() as f64
    );

    // --- 9. Write .state blob (atomic: .tmp → rename) ---
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("Cannot create output dir: {}", out_dir.display()))?;

    let state_bytes = shard.to_bytes();
    atomic_write(out_dir.join("shard.state"), &state_bytes)?;
    println!(
        "[baker] ✓ Written: shard.state ({:.1} MB)",
        state_bytes.len() as f64 / 1_048_576.0
    );

    // --- 10. Write .axons blob (geometry: tip_x, tip_y, tip_z, length per axon) ---
    let axons_bytes = serialize_axons(&axons);
    atomic_write(out_dir.join("shard.axons"), &axons_bytes)?;
    println!(
        "[baker] ✓ Written: shard.axons ({:.1} MB)",
        axons_bytes.len() as f64 / 1_048_576.0
    );

    println!("[baker] Done.");
    Ok(())
}

/// Формат файла `.axons`:
/// - Header: `AxonsFileHeader` (8 bytes)
/// - Array of `total_axons`:
///   - tip_x: u16, tip_y: u16, tip_z: u16
///   - length: u32 (N)
///   - segments: [u32; N] (PackedPositions)
fn serialize_axons(axons: &[bake::axon_growth::GrownAxon]) -> Vec<u8> {
    // Оцениваем размер: 8 байт заголовок + для каждого (10 байт заголовок + 4 байта на сегмент)
    let estimated_size = 8 + axons.iter().map(|ax| 10 + ax.segments.len() * 4).sum::<usize>();
    let mut out = Vec::with_capacity(estimated_size);
    let header = genesis_core::layout::AxonsFileHeader::new(axons.len() as u32);
    out.extend_from_slice(header.as_bytes());
    for ax in axons {
        out.extend_from_slice(&(ax.tip_x as u16).to_le_bytes());
        out.extend_from_slice(&(ax.tip_y as u16).to_le_bytes());
        out.extend_from_slice(&(ax.tip_z as u16).to_le_bytes());
        
        out.extend_from_slice(&(ax.segments.len() as u32).to_le_bytes());
        for &seg in &ax.segments {
            out.extend_from_slice(&seg.to_le_bytes());
        }
    }
    out
}

/// Атомарная запись: пишем в .tmp, потом rename. Защита от краша.
fn atomic_write(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data).with_context(|| format!("Cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("Cannot rename .tmp → {}", path.display()))?;
    Ok(())
}


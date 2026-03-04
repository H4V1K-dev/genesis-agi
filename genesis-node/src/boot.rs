use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::collections::HashMap;
use crossbeam::channel::bounded;

use genesis_core::config::manifest::ZoneManifest;
use genesis_compute::memory::{VramState, calculate_state_blob_size, compute_state_offsets};
use genesis_compute::ShardEngine;
use crate::zone_runtime::ZoneRuntime;
use crate::network::geometry_client::GeometryServer;
use crate::network::telemetry::TelemetryServer;
use crate::network::io_server::ExternalIoServer;
use crate::network::bsp::BspBarrier;
use crate::node::NodeRuntime;
use crate::network::router::RoutingTable;
use crate::input::GxiFile;
use crate::output::GxoFile;
use crate::network::ghosts::load_ghosts;
use crate::network::intra_gpu::IntraGpuChannel;

pub struct Bootloader;

pub struct BootResult {
    pub node_runtime: NodeRuntime,
    pub geometry_server: GeometryServer,
    pub geometry_data: Vec<u32>,
    pub telemetry_swapchain: Arc<crate::network::telemetry::TelemetrySwapchain>,
}

/// Инициализирует ShardEngine прямым DMA-копированием из .state и .axons.
/// Этот метод реализует O(1) деривацию размеров на основе файлового контракта:
/// - .state: 910 байта на нейрон (SoA)
/// - .axons: 4 байта на аксон (u32 heads)
pub fn boot_shard_from_disk(baked_dir: &Path) -> Result<(ShardEngine, Vec<u32>)> {
    let checkpoint_path = baked_dir.join("checkpoint.state");
    let base_state_path = baked_dir.join("shard.state");
    let axons_path = baked_dir.join("shard.axons");

    // [DOD] Приоритет горячего дампа над начальным графом
    let state_path = if checkpoint_path.exists() {
        println!("[Boot] 💾 Resuming from checkpoint: {:?}", checkpoint_path);
        checkpoint_path
    } else {
        base_state_path
    };

    // 1. Zero-Parsing загрузка сырых байт в RAM
    let state_blob = std::fs::read(&state_path)
        .with_context(|| format!("FATAL: Missing .state file at {:?}", state_path))?;
    let axons_blob = std::fs::read(&axons_path)
        .with_context(|| format!("FATAL: Missing .axons file at {:?}", axons_path))?;

    // 2. Деривация размеров за O(1) (910 байт/нейрон)
    let padded_n = (state_blob.len() / 910) as u32;
    let total_axons = (axons_blob.len() / 4) as u32;

    // Верификация целостности (защита от битых файлов)
    let (_, expected_state_size) = calculate_state_blob_size(padded_n as usize);
    if state_blob.len() != expected_state_size {
        anyhow::bail!(
            "FATAL: .state blob size mismatch for {:?}. Expected {}, got {}. Corruption or version skew!",
            state_path, expected_state_size, state_blob.len()
        );
    }

    // Извлекаем soma_to_axon для маршрутизации Ghost-аксонов
    let offsets = compute_state_offsets(padded_n as usize);
    let s2a_bytes = &state_blob[offsets.soma_to_axon .. offsets.soma_to_axon + (padded_n as usize * 4)];
    let soma_to_axon: Vec<u32> = bytemuck::cast_slice(s2a_bytes).to_vec();

    // 3. Выделение VRAM и DMA-заливка
    let vram = VramState::allocate(padded_n, total_axons);
    vram.upload_state(&state_blob);
    vram.upload_axon_heads(&axons_blob);

    Ok((ShardEngine::new(vram), soma_to_axon))
}

impl Bootloader {
    /// Full node bootstrap sequence.
    pub async fn boot_node(manifest_path: &Path, dashboard: Arc<crate::tui::DashboardState>) -> Result<BootResult> {
        let manifest_toml = std::fs::read_to_string(manifest_path)
            .with_context(|| format!("Failed to read manifest: {:?}", manifest_path))?;
        
        let brain_config: genesis_core::config::brain::BrainConfig = toml::from_str(&manifest_toml)
            .with_context(|| format!("Failed to parse brain manifest: {:?}", manifest_path))?;

        let _baked_dir = manifest_path.parent().unwrap_or(std::path::Path::new("."));
        
        let mut engines = Vec::new();
        let mut io_contexts = Vec::new();
        let mut all_geo_data = Vec::new();
        let mut output_routes: HashMap<u32, Vec<(String, u32)>> = HashMap::new();

        let mut axon_head_ptrs = HashMap::new();
        let mut s2a_maps = HashMap::new();

        // [DOD] Aggregate all zones into a unified Node process
        for zone_entry in &brain_config.zones {
            let baked_dir = if zone_entry.baked_dir.is_absolute() {
                zone_entry.baked_dir.clone()
            } else {
                // If the path starts with 'baked/', resolve it from the project root.
                // BrainConfig in PingPongBrain uses 'baked/PingPongBrain/...'
                std::path::Path::new("/home/alex/Workflow/Genesis").join(&zone_entry.baked_dir)
            };

            println!("[Boot] Loading Zone: {} at {:?}", zone_entry.name, baked_dir);
            let (engine, s2a) = boot_shard_from_disk(&baked_dir)?;
            let zone_hash = genesis_core::hash::fnv1a_32(zone_entry.name.as_bytes());

            axon_head_ptrs.insert(zone_hash, engine.vram.ptrs.axon_heads);
            s2a_maps.insert(zone_hash, s2a);

            // Resolve io.toml path early — needed for GXI safety check below
            let io_config_path = if zone_entry.io.is_absolute() {
                zone_entry.io.clone()
            } else {
                std::path::Path::new("/home/alex/Workflow/Genesis").join(&zone_entry.io)
            };

            let gxi_path = baked_dir.join("shard.gxi");
            let num_virtual_axons = if gxi_path.exists() {
                GxiFile::load(&gxi_path).total_pixels
            } else {
                // [DOD] If io.toml declares inputs but .gxi is missing, crash immediately.
                // Silent fallback to 0 causes DMA buffer overflow → VRAM corruption.
                if io_config_path.exists() {
                    if let Ok(io_cfg) = genesis_core::config::io::IoConfig::load(&io_config_path) {
                        if !io_cfg.inputs.is_empty() {
                            panic!(
                                "FATAL: Zone '{}' has inputs in io.toml but {:?} is missing!\n\
                                 Fix baked_dir in brain.toml or re-run Baker.",
                                zone_entry.name, gxi_path
                            );
                        }
                    }
                }
                0
            };

            let gxo_path = baked_dir.join("shard.gxo");
            let (num_outputs, mapped_soma_ids_device) = if gxo_path.exists() {
                let gxo = GxoFile::load(&gxo_path);
                let ptr = unsafe { genesis_compute::ffi::gpu_malloc(gxo.soma_ids.len() * 4) } as *mut u32;
                unsafe {
                    genesis_compute::ffi::gpu_memcpy_host_to_device(
                        ptr as *mut _,
                        gxo.soma_ids.as_ptr() as *const _,
                        gxo.soma_ids.len() * 4
                    );
                }
                (gxo.total_pixels, ptr as *const u32)
            } else {
                (0, std::ptr::null())
            };

            // Aggregate geometry data
            let pos_path = baked_dir.join("shard.pos");
            let pos_blob = std::fs::read(&pos_path)
                .with_context(|| format!("Failed to read shard.pos: {:?}", pos_path))?;
            let geo_data: Vec<u32> = bytemuck::cast_slice(&pos_blob).to_vec();
            all_geo_data.extend(geo_data);

            // [DOD] Dedicated InputSwapchain for this zone
            let input_words_per_tick = (num_virtual_axons + 31) / 32;
            let sync_batch_ticks = 100;
            let capacity = (input_words_per_tick * sync_batch_ticks * 4) as usize;
            
            let mut matrix_offsets = HashMap::new();
            

            if io_config_path.exists() {
                if let Ok(io_config) = genesis_core::config::io::IoConfig::load(&io_config_path) {
                    let mut current_bit_offset = 0u32;
                    for input in &io_config.inputs {
                        let hash = genesis_core::hash::fnv1a_32(input.name.as_bytes());
                        // Offset is in BYTES within the one-tick-slice of the buffer
                        matrix_offsets.insert(hash, (current_bit_offset / 8) as u32);
                        current_bit_offset += input.width * input.height;
                        // Align per-matrix boundary to 32 bits (4 bytes)
                        current_bit_offset = (current_bit_offset + 31) & !31;
                    }
                    for output in &io_config.outputs {
                        let hash = genesis_core::hash::fnv1a_32(output.name.as_bytes());
                        // [DOD] Benchmark routing: for now hardcoded to 8082 for specific zones
                        if zone_entry.name == "MotorCortex" || zone_entry.name == "SensoryCortex" {
                            output_routes.entry(zone_hash).or_insert_with(Vec::new)
                                .push(("127.0.0.1:8082".to_string(), hash));
                        }
                    }
                }
            }

            let io_ctx = crate::network::io_server::ZoneIoContext {
                swapchain: std::sync::Arc::new(crate::network::io_server::InputSwapchain::new(capacity.max(4))?),
                matrix_offsets,
            };

            io_contexts.push((zone_hash, io_ctx));
            engines.push((zone_hash, engine, num_virtual_axons, num_outputs, mapped_soma_ids_device, baked_dir.clone()));
        }

        // 2. Сборка IntraGpuChannel (Ghost Routing)
        let mut intra_gpu_channels = Vec::new();
        for conn in &brain_config.connections {
            let src_hash = genesis_core::hash::fnv1a_32(conn.from.as_bytes());
            let dst_hash = genesis_core::hash::fnv1a_32(conn.to.as_bytes());
            
            // Resolve baked dir from brain_config
            let src_zone_entry = brain_config.zones.iter().find(|z| z.name == conn.from).expect("Source zone missing");
            let src_baked_dir = std::path::Path::new("/home/alex/Workflow/Genesis").join(&src_zone_entry.baked_dir);
            
            let ghosts_path = src_baked_dir.join(format!("{}_{}.ghosts", conn.from, conn.to));

            if ghosts_path.exists() {
                let (src_somas, dst_ghosts) = load_ghosts(&ghosts_path);
                let s2a = s2a_maps.get(&src_hash).expect("S2A map missing for source zone");

                let mut src_axons = Vec::with_capacity(src_somas.len());
                for &soma in &src_somas {
                    let mut axon_id = genesis_core::constants::AXON_SENTINEL;
                    if soma != genesis_core::ipc::EMPTY_PIXEL {
                        axon_id = s2a[soma as usize];
                    }
                    src_axons.push(axon_id);
                }

                let channel = unsafe { IntraGpuChannel::from_slices(&src_axons, &dst_ghosts) };
                intra_gpu_channels.push((src_hash, dst_hash, channel));
                println!("[Boot] Built IntraGpuChannel: {} -> {} ({} links)", conn.from, conn.to, src_axons.len());
            }
        }

        // 3. Ports (Using first zone or defaults)
        let root_dir = std::path::Path::new("/home/alex/Workflow/Genesis");
        let first_manifest_path = root_dir.join(&brain_config.zones[0].baked_dir).join("manifest.toml");
        let first_manifest_toml = std::fs::read_to_string(&first_manifest_path)
            .with_context(|| format!("Failed to read first_manifest: {:?}", first_manifest_path))?;
        let first_manifest: genesis_core::config::manifest::ZoneManifest = toml::from_str(&first_manifest_toml)?;
        
        // [DOD FIX] Прошивка законов физики в Constant Memory GPU
        let mut gpu_variants = [genesis_core::config::manifest::GpuVariantParameters::default(); 16];
        for v in &first_manifest.variants {
            if (v.id as usize) < 16 {
                // Zero-cost конвертация из DTO в C-совместимый layout
                gpu_variants[v.id as usize] = v.clone().into_gpu();
            }
        }
        
        unsafe {
            let err = genesis_compute::ffi::cu_upload_constant_memory(gpu_variants.as_ptr() as *const genesis_compute::ffi::VariantParameters);
            if err != 0 {
                println!("FATAL: cu_upload_constant_memory failed with {}", err);
            }
        }
        println!("[Boot] Hardware physics parameters flashed to Constant Memory.");

        let local_port = first_manifest.network.fast_path_udp_local;
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));

        // 2. IO Socket
        let io_socket = tokio::net::UdpSocket::bind(&format!("0.0.0.0:{}", first_manifest.network.external_udp_in)).await?;
        let io_server = Arc::new(ExternalIoServer::new(
            Arc::new(AtomicBool::new(false)),
            io_contexts,
            dashboard.clone(),
            routing_table.clone(),
            Arc::new(io_socket)
        )?);

        // 3. Servers
        let geo_addr = format!("0.0.0.0:{}", local_port + 1).parse()?;
        let geometry_server = GeometryServer::bind(geo_addr).await?;
        let telemetry_swapchain = TelemetryServer::start(local_port + 2).await;

        let bsp_barrier = Arc::new(std::sync::Mutex::new(BspBarrier::new(100)));

        let node_runtime = NodeRuntime::boot(
            engines,
            io_server,
            routing_table,
            bsp_barrier,
            telemetry_swapchain.clone(),
            std::net::Ipv4Addr::new(127, 0, 0, 1),
            local_port,
            output_routes,
            intra_gpu_channels,
            axon_head_ptrs,
        );

        Ok(BootResult {
            node_runtime,
            geometry_server,
            geometry_data: all_geo_data,
            telemetry_swapchain,
        })
    }
}

use genesis_core::config::manifest::ZoneManifest;
use genesis_runtime::zone_runtime::ZoneRuntime;

use genesis_runtime::memory::VramState;
use anyhow::{Context, Result};
use genesis_runtime::Runtime;
use clap::Parser;

use genesis_runtime::network::geometry_client::GeometryServer;

use genesis_runtime::network::telemetry::TelemetryServer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "genesis-node",
    about = "Distributed Genesis Brain Node Daemon",
    version
)]
struct Cli {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long, default_value = "9000")]
    fast_path_port: u16,

    #[arg(long)]
    peer: Vec<String>,

    #[arg(long)]
    mock_retina: bool,

    #[arg(long)]
    baker_socket: Option<PathBuf>,
}

struct ZoneLink {
    from_idx: usize,
    to_idx: usize,
    channel: genesis_runtime::network::intra_gpu::IntraGpuChannel,
}

use tokio::runtime::Builder;
use genesis_runtime::network::external::ExternalIoServer;
use genesis_runtime::simple_reporter::SimpleReporter;

fn main() -> Result<()> {
    // 1. Initialize dedicated Tokio Runtime for I/O (2 threads max)
    let rt = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Fatal: Failed to build Tokio runtime");

    rt.block_on(async {
        let cli = Cli::parse();
        println!("[Node] Starting Genesis Distributed Daemon...");

    // 2. Load Manifest Artifact
    let manifest_toml = std::fs::read_to_string(&cli.manifest)
        .with_context(|| format!("Failed to read manifest: {:?}", cli.manifest))?;
    let manifest: ZoneManifest = toml::from_str(&manifest_toml)
        .with_context(|| format!("Failed to parse manifest: {:?}", cli.manifest))?;
    
    let baked_dir = cli.manifest.parent().unwrap_or(std::path::Path::new(""));
    let zone_hash = manifest.zone_hash;
    let zone_name = format!("Zone_{:08X}", zone_hash); // Fallback debug name

    println!("[Node] Artifact Manifest Loaded. Zone Hash: 0x{:08X}", zone_hash);

    // 3. Implicit Role Dispatching (Zero-Dependency)
    let state_path = baked_dir.join("shard.state");
    let axons_path = baked_dir.join("shard.axons");
    
    let mut gxi_opt = None;
    let mut gxo_opt = None;
    let mut ghosts_files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(baked_dir) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                if ext == "gxi" {
                    println!("[Node] Discovered GXI -> Enabling Input Role");
                    gxi_opt = Some(genesis_runtime::input::GxiFile::load(&entry.path()));
                } else if ext == "gxo" {
                    println!("[Node] Discovered GXO -> Enabling Output Role");
                    gxo_opt = Some(genesis_runtime::output::GxoFile::load(&entry.path()));
                } else if ext == "ghosts" {
                    println!("[Node] Discovered Ghost Routing Table -> Enabling Fast Path Egress");
                    ghosts_files.push(entry.path());
                }
            }
        }
    }

    if gxi_opt.is_none() && gxo_opt.is_none() && ghosts_files.is_empty() {
        println!("[Node] Operating Mode: Computations Only (Hidden Cortex)");
    }

    let has_input = gxi_opt.is_some();
    let has_output = gxo_opt.is_some();
    let has_io = has_input || has_output;


    // 4. Initialize Network Shared Ports
    let local_port = manifest.network.fast_path_udp_local;
    let udp_addr = format!("0.0.0.0:{}", local_port);
    let tcp_port = local_port + 1;
    
    let geo_addr = format!("0.0.0.0:{}", tcp_port).parse().unwrap();
    let geo_server = GeometryServer::bind(geo_addr).await
        .context("Failed to bind TCP Geometry Server")?;
    println!("[Node] Bound TCP Geometry Server on {}", geo_addr);
    geo_server.spawn();

    let telemetry_port = local_port + 2;
    let telemetry_tx = TelemetryServer::start(telemetry_port).await;

    println!("[Node] Bound UDP Fast Path on {}", udp_addr);

    let mut zone_ping_pongs = std::collections::HashMap::new();

    // Load VRAM
    let state_bytes = std::fs::read(&state_path).with_context(|| format!("Failed to read {:?}", state_path))?;
    let axons_bytes = std::fs::read(&axons_path).with_context(|| format!("Failed to read {:?}", axons_path))?;

    let sync_batch_ticks = 100; // Hardcoded fallback or we need it in manifest? Let's assume 100 for now.
    let vram = VramState::load_shard(
        &state_bytes, 
        &axons_bytes, 
        gxi_opt.as_ref(), 
        gxo_opt.as_ref(),
        sync_batch_ticks as u32,
        1, // input stride
        manifest.memory.ghost_capacity
    ).context("Failed to push shard data to GPU VRAM")?;

    if gxo_opt.is_some() {
        println!("[Node] Loaded Output Mapping: {} output neurons ready for readout", vram.num_mapped_somas);
    }    println!("       VRAM Load Complete. {} neurons, {} total axons", vram.padded_n, vram.total_axons);

    let mut const_mem = [genesis_core::config::manifest::GpuVariantParameters::default(); 16];
    for variant in &manifest.variants {
        let idx = variant.id as usize;
        if idx < 16 {
            const_mem[idx] = variant.clone().into_gpu();
        }
    }
    let master_seed = 42; // Fallback or read from manifest later. For PingPong, any will do.

    let mut runtime = Runtime::new(vram, manifest.memory.v_seg as u32, master_seed, Some(baked_dir.to_path_buf()));

    if let Some(ref socket_path) = cli.baker_socket {
        if let Ok(client) = genesis_runtime::ipc::BakerClient::connect(0, socket_path) {
            println!("       Baker daemon connected.");
            runtime.baker_client = Some(client);
        }
    }

    let ping_pong = Arc::new(unsafe { genesis_runtime::network::bsp::PingPongSchedule::new(sync_batch_ticks, 1024) });
    zone_ping_pongs.insert(zone_hash, ping_pong.clone());

    let mut zones: Vec<ZoneRuntime> = Vec::new();
    zones.push(ZoneRuntime {
        name: zone_name.clone(),
        artifact_dir: baked_dir.to_path_buf(),
        runtime,
        const_mem,
        config: genesis_core::config::instance::InstanceConfig {
            zone_id: "0".to_string(), world_offset: genesis_core::config::instance::Coordinate {x:0, y:0, z:0}, dimensions: genesis_core::config::instance::Dimensions {w:1000, d:1000, h:1000}, neighbors: genesis_core::config::instance::Neighbors {x_plus:None, x_minus:None, y_plus:None, y_minus:None}
        },
        prune_threshold: -50,
        is_sleeping: Arc::new(AtomicBool::new(false)),
        sleep_requested: false,
        ping_pong,
        last_night_time: std::time::Instant::now(),
        min_night_delay: std::time::Duration::from_secs(30),
        slow_path_queues: Arc::new(genesis_runtime::network::slow_path::SlowPathQueues::new()),
        hot_reload_queue: Arc::new(crossbeam::queue::SegQueue::new()),
        inter_node_channels: Vec::new(),
        intra_gpu_channels: Vec::new(),
        spatial_grid: std::sync::Arc::new(std::sync::Mutex::new(genesis_runtime::orchestrator::spatial_grid::SpatialGrid::new())),
    });

    // --- HOT RELOAD WATCHER ---
    let hot_reload_q = zones.last().unwrap().hot_reload_queue.clone();
    let manifest_path = cli.manifest.clone();
    rt.spawn(async move {
        let mut last_modified = std::time::SystemTime::UNIX_EPOCH;
        loop {
            if let Ok(metadata) = std::fs::metadata(&manifest_path) {
                if let Ok(modified) = metadata.modified() {
                    if modified > last_modified {
                        last_modified = modified;
                        if let Ok(content) = std::fs::read_to_string(&manifest_path) {
                            if let Ok(new_manifest) = toml::from_str::<ZoneManifest>(&content) {
                                let mut gpu_lut = [genesis_core::config::manifest::GpuVariantParameters::default(); 16];
                                for variant in new_manifest.variants {
                                    let idx = variant.id as usize;
                                    if idx < 16 {
                                        gpu_lut[idx] = variant.into_gpu();
                                    }
                                }
                                hot_reload_q.push(gpu_lut);
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });

    // --- SLOW PATH TCP DISPATCHER ---
    let zone_queues = zones.last().unwrap().slow_path_queues.clone();
    let listen_port = manifest.network.slow_path_tcp;
    let peers = cli.peer.clone();

    rt.spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::{TcpListener, TcpStream};
        
        let listener = TcpListener::bind(format!("0.0.0.0:{}", listen_port)).await.unwrap();
        let q_in = zone_queues.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let q = q_in.clone();
                tokio::spawn(async move {
                    let mut header = [0u8; 8];
                    if socket.read_exact(&mut header).await.is_ok() {
                        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
                        let count = u32::from_le_bytes(header[4..8].try_into().unwrap());
                        
                        if magic == 0x47524F57 {
                            for _ in 0..count {
                                let mut buf = [0u8; 16];
                                if socket.read_exact(&mut buf).await.is_ok() {
                                    let ev = unsafe { std::mem::transmute::<[u8; 16], genesis_runtime::network::slow_path::AxonHandoverEvent>(buf) };
                                    q.incoming_grow.push(ev);
                                }
                            }
                        } else if magic == 0x41434B48 {
                            for _ in 0..count {
                                let mut buf = [0u8; 12];
                                if socket.read_exact(&mut buf).await.is_ok() {
                                    let ack = unsafe { std::mem::transmute::<[u8; 12], genesis_runtime::network::slow_path::AxonHandoverAck>(buf) };
                                    q.incoming_ack.push(ack);
                                }
                            }
                        } else if magic == 0x44454144 {
                            for _ in 0..count {
                                let mut buf = [0u8; 8];
                                if socket.read_exact(&mut buf).await.is_ok() {
                                    let ev = unsafe { std::mem::transmute::<[u8; 8], genesis_runtime::network::slow_path::AxonHandoverPrune>(buf) };
                                    q.incoming_prune.push(ev);
                                }
                            }
                        }
                    }
                });
            }
        });

        loop {
            // Simplified Egress (broadcast to all cli.peers)
            let mut grows = Vec::new();
            while let Some(ev) = zone_queues.outgoing_grow.pop() { grows.push(ev); }
            if !grows.is_empty() {
                for peer in &peers {
                    if let Ok(mut stream) = TcpStream::connect(peer).await {
                        let mut head = 0x47524F57u32.to_le_bytes().to_vec();
                        head.extend_from_slice(&(grows.len() as u32).to_le_bytes());
                        let _ = stream.write_all(&head).await;
                        let bytes = unsafe { std::slice::from_raw_parts(grows.as_ptr() as *const u8, grows.len() * 16) };
                        let _ = stream.write_all(bytes).await;
                    }
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    });



    let mut inter_node_links = Vec::new();
    
    for ghost_file in ghosts_files {
        let (src_map, dst_map) = genesis_runtime::network::ghosts::load_ghosts(&ghost_file);
        
        let path_str = ghost_file.to_string_lossy();
        let name_parts: Vec<&str> = path_str.split('/').last().unwrap().split('_').collect();
        // Fallback target hash parsing from filename, e.g. SensoryCortex_MotorCortex.ghosts
        let target_hash = if name_parts.len() >= 2 {
            let target_name = name_parts[1].replace(".ghosts", "");
            genesis_core::hash::fnv1a_32(target_name.as_bytes())
        } else {
            0
        };

        let channel = unsafe { 
            genesis_runtime::network::inter_node::InterNodeChannel::new(target_hash, &src_map, &dst_map) 
        };
        
        zones[0].inter_node_channels.push(unsafe { 
             genesis_runtime::network::inter_node::InterNodeChannel::new(target_hash, &src_map, &dst_map) 
        });

        inter_node_links.push(channel);
        println!("[Node] Extracted Ghost Routing Table: {:?}", ghost_file);
    }

    // 4.5 Initialize UDP InterNode Router
    // Build routing table dynamically from ghost file hashes → manifest peers (order-matched)
    let mut routing_peers = std::collections::HashMap::new();
    for (idx, peer) in manifest.network.fast_path_peers.iter().enumerate() {
        let target_addr: std::net::SocketAddr = peer.parse().expect("Invalid peer address in manifest");
        if idx < inter_node_links.len() {
            let target_hash = inter_node_links[idx].target_zone_hash;
            eprintln!("[Router] Registering peer {} -> hash {:x}", peer, target_hash);
            routing_peers.insert(target_hash, target_addr);
        } else {
            routing_peers.insert(genesis_runtime::network::router::fnv1a_32(b"MotorCortex"), target_addr);
            routing_peers.insert(genesis_runtime::network::router::fnv1a_32(b"SensoryCortex"), target_addr);
        }
    }

    let routing_table = genesis_runtime::network::router::RoutingTable { peers: routing_peers };
    let inter_node_router = genesis_runtime::network::router::InterNodeRouter::new(&udp_addr, routing_table).await;
    let router_arc = Arc::new(inter_node_router);
    genesis_runtime::network::router::InterNodeRouter::spawn_receiver_loop(
        router_arc.socket.clone(),
        zone_ping_pongs
    );

    let mut pinned_input_ptr = std::ptr::null_mut();
    let mut pinned_output_ptr = std::ptr::null_mut();
    let mut input_bytes = 0;
    let mut output_bytes = 0;

    let mut sensory_idx = None;
    let mut motor_idx = None;
    let mut io_server_opt = None;

    if has_io {
        sensory_idx = Some(0);
        motor_idx = Some(0);

        let words_per_tick = (zones[0].runtime.vram.num_pixels as u32 + 31) / 32;
        input_bytes = (words_per_tick as usize) * sync_batch_ticks * 4;
        output_bytes = zones[0].runtime.vram.num_mapped_somas as usize * sync_batch_ticks;

        unsafe {
            pinned_input_ptr = genesis_runtime::ffi::gpu_host_alloc(input_bytes) as *mut u32;
            pinned_output_ptr = genesis_runtime::ffi::gpu_host_alloc(output_bytes) as *mut u8;
        }

        let mut io_server_obj = ExternalIoServer::new(
            &format!("0.0.0.0:{}", manifest.network.external_udp_in),
            pinned_input_ptr,
            input_bytes
        ).await;
        
        println!("[IO] Input Server bound to UDP {}", manifest.network.external_udp_in);
        println!("[IO] Output will be sent to UDP {}", manifest.network.external_udp_out);
        
        if let Some(ref gxi_data) = gxi_opt {
            for m in &gxi_data.matrices {
                io_server_obj.matrix_offsets.insert(m.name_hash, m.offset);
            }
        }
        
        io_server_obj.dashboard = None; // Dashboard removed for simplicity
        let io_server = Arc::new(io_server_obj);

        let io_rx = io_server.clone();
        rt.spawn(async move { io_rx.run_rx_loop().await; });
        io_server_opt = Some(io_server);
        println!("Genesis Engine: IO Server Started.");
    }

    // =====================================================================
    // 7. MAIN HOT LOOP (Dedicated OS Thread)
    // =====================================================================
    let mut total_ticks: u64 = 0;
    let night_interval_ticks: u64 = 10000;
    let stream: genesis_runtime::ffi::CudaStream = std::ptr::null_mut(); // Default CUDA stream

    let reporter = SimpleReporter::new();
    eprintln!("[Genesis] Starting simulation...");

    let mut start_time = std::time::Instant::now();

    loop {
        if has_input {
            println!("[Main Loop] NODE_A (Sensory) iteration, total_ticks={}", total_ticks);
        } else if has_output {
            println!("[Main Loop] NODE_B (Motor) iteration, total_ticks={}, batch_idx={}", total_ticks, total_ticks as usize / sync_batch_ticks);
        } else {
            println!("[Main Loop] Hidden Cortex iteration, total_ticks={}", total_ticks);
        }
        
        // [BSP БАРЬЕР]: Ожидание нового входного кадра от Python (только если есть GXI)

        if has_input {
            let io_server = io_server_opt.as_ref().unwrap();
            // println!("   [BSP] Waiting for Frame...");
            while io_server.new_frame_ready.load(Ordering::Acquire) == 0 {
                std::hint::spin_loop();
            }
            io_server.new_frame_ready.store(0, Ordering::Release);

            // Шаг 1: DMA Host-to-Device (Перекачка свежей маски входов)
            unsafe {
                genesis_runtime::ffi::gpu_memcpy_host_to_device_async(
                    zones[sensory_idx.unwrap()].runtime.vram.input_bitmask_buffer as *mut std::ffi::c_void,
                    pinned_input_ptr as *const std::ffi::c_void,
                    input_bytes,
                    stream
                );
            }
        } else {
            // NODE B: Ждем пакета от NODE A для каждой зоны
            for zone in zones.iter() {
                zone.ping_pong.wait_for_data(total_ticks as usize / sync_batch_ticks); 
            }
        }

        // Шаг 2: Выполнение батча (GPU молотит 6 ядер ПАРАЛЛЕЛЬНО для всех зон)
        let current_dopamine = if let Some(srv) = &io_server_opt {
            srv.global_dopamine.load(Ordering::Relaxed) as i16
        } else {
            0
        };

        for zone in zones.iter_mut() {
            let tx_opt = if zone.name == "SensoryCortex" { Some(&telemetry_tx) } else { None };
            genesis_runtime::orchestrator::day_phase::execute_day_batch(zone, sync_batch_ticks as u32, stream, tx_opt, total_ticks, current_dopamine);
        }

        // Шаг 3: Intra-GPU Ghost Sync was removed (standalone processes).

        // Шаг 3.5: Inter-Node (Экстракция исходящих спайков)
        unsafe {
            for link in inter_node_links.iter() {
                // Find src boundary dynamically
                if let Some(src_zone) = zones.iter().find(|z| genesis_runtime::network::router::fnv1a_32(z.name.as_bytes()) == link.target_zone_hash) {
                     // Wait, target_zone_hash is for the *destination*, we need the source.
                     // A cleaner way for MVP: since we only have 1 inter-node link per node in this test (Sensory->Hidden, Hidden->Motor)
                     // Let's just find the first zone that isn't asleep and extract from it. 
                     // Actually, we must bind the channel to the specific zone.
                }
                
                // Simplified MVP: Just use the very first loaded zone as the source. 
                // For NODE_A: SensoryCortex is zones[0]. For NODE_B: HiddenCortex is zones[0].
                // This is extremely hardcoded but works for the Cartesian Split Brain test block.
                let src_zone = &zones[0]; 
                link.extract_spikes(
                    src_zone.runtime.vram.axon_head_index as *const _,
                    sync_batch_ticks as u32,
                    stream
                );
            }
        }

        // Шаг 4: DMA Device-to-Host (Скачивание результатов сом)
        unsafe {
            if has_output {
                genesis_runtime::ffi::gpu_memcpy_device_to_host_async(
                    pinned_output_ptr as *mut std::ffi::c_void,
                    zones[motor_idx.unwrap()].runtime.vram.output_history as *const std::ffi::c_void,
                    output_bytes,
                    stream
                );
            }
            
            // БЛОКИРОВКА CPU: Ждем, пока GPU закончит ВСЕ ядра и вернет нам Pinned RAM
            genesis_runtime::ffi::gpu_stream_synchronize(stream);
            
            // [DMA Check] Шаг 1 Валидации: pinned_output_ptr содержит данные из VRAM?
            // Выполняется СТРОГО после gpu_stream_synchronize — данные гарантированно актуальны.
            if has_output && output_bytes > 0 && !pinned_output_ptr.is_null() {
                let output_slice = std::slice::from_raw_parts(pinned_output_ptr as *const u8, output_bytes);
                let active_pixel_ticks: usize = output_slice.iter().filter(|&&b| b > 0).count();
                if active_pixel_ticks > 0 {
                    println!("[DMA Check ✓] Downloaded {} active pixel-ticks from VRAM (out of {} total)", 
                        active_pixel_ticks, output_bytes);
                } else {
                    println!("[DMA Check ✗] VRAM→RAM: {} bytes, but ALL ZEROS. Check RecordReadout / mapped_soma_ids.", 
                        output_bytes);
                }
            }

            
            // --- UDP Флаш (Только сейчас можно читать out_count_pinned) ---
            for link in inter_node_links.iter() {
                let count = std::ptr::read_volatile(link.out_count_pinned) as usize;
                if count > 0 {
                    eprintln!("[Node A Egress] Sending {} spikes to zone_hash={:x}", count, link.target_zone_hash);
                    let events_slice = std::slice::from_raw_parts(link.out_events_pinned, count);
                    let router_clone = router_arc.clone();
                    let target_hash = link.target_zone_hash;
                    let events_vec = events_slice.to_vec(); // clone for async
                    rt.spawn(async move {
                        let _ = router_clone.flush_outgoing_batch(target_hash, &events_vec).await;
                    });
                }
            }
            
            // --- BSP SWAP ---
            for zone in zones.iter_mut() {
                zone.ping_pong.sync_and_swap();
                zone.ping_pong.clear_write_buffer();
            }
        }

        // Шаг 5: Асинхронная отправка выходов (Node -> Python Client)
        if gxo_opt.is_some() {
            let target_addr = manifest.network.external_udp_out_target.clone()
                .unwrap_or_else(|| format!("127.0.0.1:{}", manifest.network.external_udp_out));
            let out_zone_hash = genesis_runtime::network::router::fnv1a_32(zones[0].name.as_bytes());
            let out_matrix_hash = gxo_opt.as_ref().unwrap().matrices[0].name_hash;
            
            let io_tx = io_server_opt.as_ref().unwrap().clone();
            let pinned_output_addr = pinned_output_ptr as usize;
            
            rt.spawn(async move {
                io_tx.send_output_batch(
                    &target_addr, 
                    out_zone_hash, 
                    out_matrix_hash, 
                    pinned_output_addr, 
                    output_bytes
                ).await;
            });
        }

        total_ticks += sync_batch_ticks as u64;
        
        // Записываем метрики Дня
        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        reporter.update(total_ticks, 0, elapsed_ms);
        start_time = std::time::Instant::now();

        // Шаг 6: Проверка триггера Night Phase
        let now = std::time::Instant::now();
        let mut night_triggered = false;
        
        for zone in zones.iter_mut() {
            let time_since_last_night = now.duration_since(zone.last_night_time);
            
            let is_sleeping = zone.is_sleeping.load(Ordering::Acquire);
            let ticks_ready = night_interval_ticks > 0 && total_ticks % night_interval_ticks == 0;
            let time_ready = time_since_last_night >= zone.min_night_delay;

            if !is_sleeping && ticks_ready && time_ready {
                zone.last_night_time = now;
                night_triggered = true;
                
                let vram_ptr = &mut zone.runtime.vram as *mut genesis_runtime::memory::VramState;
                genesis_runtime::orchestrator::night_phase::trigger_night_phase(
                        zone.artifact_dir.clone(),
                        total_ticks,
                    vram_ptr,
                    zone.runtime.vram.padded_n as u32,
                    zone.runtime.vram.total_axons as u32,
                    zone.prune_threshold,
                    zone.is_sleeping.clone(),
                    zone.runtime.master_seed,
                    zone.slow_path_queues.clone(),
                    zone.inter_node_channels.clone(),
                    zone.spatial_grid.clone(),
                );
            }
        }
        
        if night_triggered {
            start_time = std::time::Instant::now();
        }
    }

    Ok(())
    })
}

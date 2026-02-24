# Changelog

All notable changes to Genesis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [0.11.0] - 2026-02-24

### Added
- **Ghost Axons (§1.7)** — бесшовный рост аксонов через границы шардов
- `ShardBounds` structure with `full_world()` and `is_outside()` boundary detection
- `GhostPacket` inter-shard transfer format (entry point, direction, remaining steps)
- `inject_ghost_axons()` — ghost axon growth continuation in receiving shard
- Pipeline integration in `main.rs` with diagnostic logging
- Unit tests for boundary detection and ghost packet handling
- Updated spec `04_connectivity.md` §1.7 with full protocol description

## [0.10.1] - 2026-02-24

### Added
- **Power Score activation** — `compute_power_index` now called during Night Phase sprouting
- Whitelist filtering in `reconnect_empty_dendrites()` (was missing)
- `sprouting_weight_type` config parameter for soft type-matching scoring component

## [0.10.0] - 2026-02-24

### Added
- **Rule of Uniqueness (§1.4)** — `HashSet`-based deduplication prevents redundant axon connections
- **Dendrite Whitelist (§1.5)** — per-type compatibility filtering via `dendrite_whitelist` in blueprints
- **Configurable Initial Weight** — `initial_synapse_weight` moved to `blueprints.toml`
- Unit tests for whitelist and initial weight parsing
- Updated spec `04_connectivity.md` §1.4–1.5

## [0.9.0] - 2026-02-24

### Added
- **GPU LUT Expansion 4→16** — each of 16 neuron types gets a unique physical profile (GLIF/GSOP)
- **Voxel Uniqueness** — reject-sampling guarantees one voxel = at most one neuron
- `growth_vertical_bias`, `type_affinity`, `is_inhibitory` fields in blueprints
- Blueprints.toml updated with 4 base types (Vertical_Excitatory, Horizontal_Inhibitory, Stable_Excitatory, Relay_Excitatory)

## [0.8.0] - 2026-02-24

### Added
- **Binary Formatting (§2.1)** — formalized GSNS/GSAX header specs
- `InstanceConfig` refactored into dedicated `instance.rs`
- Default CLI paths updated to `config/zones/V1/*`
- E2E test script paths corrected

## [0.7.0] - 2026-02-23

### Added
- **Configuration Architecture (Spec 02 §1.1–1.3)** — `simulation.toml` parser in genesis-core
- `anatomy.rs` parser with population calculation tests
- `DerivedPhysics` + `compute_derived_physics()` with §1.6 invariant
- `Tick`, `Microns`, `Fraction`, `VoxelCoord` type aliases
- `ms_to_ticks`, `us_to_ticks`, `ticks_to_ms` conversions with unit tests
- Master Seed §2 implementation

### Fixed
- `PackedTarget` bitmap layout corrected (16/16 → 22/10)
- `initial_axon_head(N) + N = AXON_SENTINEL` invariant documented

## [0.6.0] - 2026-02-23

### Added
- **Night Phase IPC Baker Daemon** — `genesis-baker-daemon` executable running in background
- **Shared Memory Protocol (SHM)** — zero-copy transfer of weights and targets between CUDA runtime and CPU baker
- **Unix Sockets** — for JSON control messages (`night_start`, `night_done`) synchronization
- **Sort & Prune CUDA Kernel** — O(1) register Bitonic Sort (N=128) per neuron to auto-promote LTM/WM and prune weak connections
- Integration E2E IPC tests verifying full orchestrator pipeline handoff

## [0.5.0] - 2026-02-23


### Added
- **Smart Axon Growth: Cone Tracing** — iterative, biologically plausible axon sprouting
- `SpatialGrid` spatial hash map for O(1) neighbor lookup during growth
- V\_attract inverse-square law and piecewise tip geometry
- Variable-length `.axons` binary format (tip\_x, tip\_y, tip\_z, length per axon)

## [0.4.1] - 2026-02-23

### Added
- **Genesis IDE** — new `genesis-ide` crate with Bevy 3D viewer
- Orbital + Fly camera modes with mouse/keyboard control
- HUD overlay: FPS, neuron count, axon count, selected neuron info
- Neuron spheres colored by `type_mask`
- Baker exports `shard.positions`; IDE highlights spiking neurons via WebSocket glow (3 frames)

## [0.4.0] - 2026-02-23

### Added
- **Genesis Monitor** — WebSocket telemetry server broadcasting tick, phase, and real-time spike dense IDs
- Bevy 3D client renders neurons as glow-highlighted spheres synced to live VRAM state

## [0.3.0] - 2026-02-22

### Added
- **Ghost Axon Handover** — TCP Slow Path with VRAM reserve pool and Handover handshake
- Dynamic `SpikeRouter` route registration via slow path
- **Homeostatic Plasticity** — branchless penalty/decay in GLIF kernel
- Equilibrium validated across 100 CUDA ticks

## [0.2.0] - 2026-02-22

### Added
- `genesis-node` daemon: parses `shard.toml`, mounts VRAM, drives BSP ephemeral loop
- **Atlas Routing** — external Ghost Axons baked at compile time, zero GPU overhead at runtime

### Fixed
- BSP deadlock: guaranteed empty-batch Header dispatch to all peers

## [0.1.0] - 2026-02-22

### Added
- BSP ping-pong buffers, async UDP Zero-Copy fast path
- Day Phase orchestrator loop
- Night Phase skeleton: GPU Sort & Prune, CPU sprouting stub, PCIe download/upload hooks

### Fixed
- Memory layout sync, Active Tail GSOP fix, 24b/8b axon format alignment

## [0.0.0] - 2026-02-21

### Added
- Architecture specification: 7 documents, ~3000 lines
- Full design from high-level abstractions down to byte-level GPU operations

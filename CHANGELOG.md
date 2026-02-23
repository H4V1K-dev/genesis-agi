# Changelog

All notable changes to Genesis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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

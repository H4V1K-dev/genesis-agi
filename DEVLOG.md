# Genesis — Devlog

## Roadmap

- [x] Architecture specification
- [x] Logic audit (6 scenarios)
- [x] `genesis-core` — types, constants, SoA layout
- [x] `genesis-baker` — TOML → `.state` / `.axons`
- [x] `genesis-runtime` — CUDA kernels + host orchestrator
- [x] Distributed runtime — BSP, UDP Fast Path, Day/Night Phase
- [x] Slow Path TCP — Ghost Axon Handover, Geometry Requests
- [x] Homeostatic Plasticity — branchless GLIF kernel
- [x] Genesis Monitor — WebSocket telemetry server
- [x] Genesis IDE — Bevy 3D viewer, camera, spike glow
- [x] Smart Axon Growth — Cone Tracing, SpatialGrid, piecewise geometry
- [x] Night Phase CPU Sprouting — full reconnect (not a stub)
- [x] End-to-end: baker → runtime → GPU
- [x] Configuration Architecture — parsers, type system, derived physics
- [x] Connectivity Audit §1 — growth differentiation, whitelist, ghost axons
- [ ] Connectivity Audit §2 — plasticity, GSOP, inertia curves
- [ ] First learning experiment
- [ ] V 1.0.0 Release

---

### [2026-02-24] V 0.11.0 — Connectivity Deep Audit

Full code-spec audit of `04_connectivity.md` §1.1–1.7:
- H/V axon growth differentiation, type_affinity, is_inhibitory (Dale's Law)
- Rule of Uniqueness (HashSet dedup), dendrite whitelist, configurable initial weight
- Power Score activation (was dead code), sprouting_weight_type soft scoring
- Ghost Axons pipeline: ShardBounds, GhostPacket, inject_ghost_axons, crossing detection
- All tests green (34 genesis-core + 4 ghost axon tests)

### [2026-02-23] V 0.7.0–0.8.0 — Configuration Foundation

- simulation.toml parser, anatomy.rs, InstanceConfig refactor
- Type system: Tick, Microns, Fraction, VoxelCoord
- DerivedPhysics, time conversions, Master Seed, PackedTarget 22/10 fix
- Binary formatting spec, CLI paths cleanup, LUT expansion 4→16

### [2026-02-22] V 0.1.0–0.3.0 — Runtime Backbone

- BSP ping-pong, UDP zero-copy, Day/Night orchestrator
- Atlas Routing, genesis-node daemon
- Ghost Axon Handover TCP, Homeostatic Plasticity

### [2026-02-21] V 0.0.0 — Hello World

First commit. Architecture specification complete. 7 docs, ~3000 lines.
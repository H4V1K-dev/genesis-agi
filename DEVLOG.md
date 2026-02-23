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
- [ ] First learning experiment
- [ ] V 1.0.0 Release

---

### [2026-02-21] V 0.0.0 — Hello World

First commit. Architecture specification complete. 7 docs, ~3000 lines.
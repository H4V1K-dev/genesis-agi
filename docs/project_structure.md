# Структура Проекта Genesis

Rust Workspace с тремя крейтами, соответствующими архитектурным слоям из спецификации.

```
Genesis/
├── Cargo.toml                  ← workspace root
├── docs/
│   ├── specs/                  ← архитектурные спецификации (7 файлов)
│   └── project_structure.md    ← этот файл
│
├── genesis-core/               ← общие типы, константы, SoA layout
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── types.rs            — PackedPosition, type aliases (u32, i16, i32)
│       ├── constants.rs        — AXON_SENTINEL, MAX_DENDRITE_SLOTS, PROPAGATION_LENGTH...
│       └── layout.rs           — SoA структуры, padded_n, Columnar Layout helpers
│
├── genesis-baker/              ← TOML → бинарные блобы (.state, .axons) — CPU only
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs             — CLI: `baker compile <anatomy.toml> <blueprints.toml>`
│       ├── parser/             — разбор TOML конфигов
│       │   ├── mod.rs
│       │   ├── anatomy.rs      — зоны, нейронные группы, топология
│       │   └── blueprints.rs   — нейронные типы, параметры, sprouting weights
│       ├── validator/          — проверка инвариантов
│       │   ├── mod.rs
│       │   └── checks.rs       — Sentinel assert, inertia_lut×potentiation >= 1, ...
│       └── bake/               — сборка бинарных блобов
│           ├── mod.rs
│           ├── layout.rs       — SoA упаковка (Columnar, padded_n, warp alignment)
│           └── sprouting.rs    — compute_power_index, sprouting_score
│
├── genesis-runtime/            ← оркестратор + CUDA ядра (Day/Night Cycle)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── orchestrator/       — управление Day/Night Cycle, триггеры сна
│       │   ├── mod.rs
│       │   ├── day_phase.rs    — запуск GPU батчей, BSP барьер
│       │   └── night_phase.rs  — Maintenance Pipeline (5 шагов)
│       ├── network/            — BSP синхронизация, TCP/Ring Buffer
│       │   ├── mod.rs
│       │   ├── bsp.rs          — Strict BSP, сетевой барьер
│       │   └── ring_buffer.rs  — SpikeSchedule, Ping-Pong Double Buffering
│       └── kernels/            — CUDA .cu файлы (Day Phase GPU ядра)
│           ├── mod.rs          — FFI биндинги
│           ├── update_neurons.cu
│           ├── propagate_axons.cu
│           ├── apply_gsop.cu
│           ├── apply_spike_batch.cu
│           ├── inject_inputs.cu
│           └── record_outputs.cu
```

## Зависимости между крейтами

```
genesis-core  ←─── genesis-baker
genesis-core  ←─── genesis-runtime
```

`genesis-baker` и `genesis-runtime` не зависят друг от друга.  
Общий контракт данных (блобы `.state` / `.axons`) — файловый обмен.

## Соответствие спецификации

| Крейт | Спека |
|---|---|
| `genesis-core` | `07_gpu_runtime.md` §1 (SoA Layout, VariantParameters) |
| `genesis-baker` | `07_gpu_runtime.md` §2.2.4 (Baking), `02_configuration.md`, `04_connectivity.md` §1.6.1 |
| `genesis-runtime` | `05_signal_physics.md`, `06_distributed.md`, `07_gpu_runtime.md` §2 |

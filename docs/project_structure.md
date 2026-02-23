# Структура Проекта Genesis

Rust Workspace с четырьмя крейтами, соответствующими архитектурным слоям из спецификации.

```
Genesis/
├── Cargo.toml                  ← workspace root
├── Cargo.lock
├── config/                     ← TOML конфиги (анатомия, blueprints)
│   ├── anatomy.toml
│   └── blueprints.toml
├── baked/                      ← бинарные блобы, сгенерированные baker
├── docs/
│   ├── specs/                  ← архитектурные спецификации (8 файлов)
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
│       ├── lib.rs
│       ├── main.rs             — CLI: `baker compile <anatomy.toml> <blueprints.toml>`
│       ├── parser/             — разбор TOML конфигов
│       │   ├── mod.rs
│       │   ├── anatomy.rs      — зоны, нейронные группы, топология
│       │   ├── blueprints.rs   — нейронные типы, параметры, sprouting weights
│       │   ├── io.rs           — чтение/запись файлов
│       │   └── simulation.rs   — параметры симуляции
│       ├── validator/          — проверка инвариантов
│       │   ├── mod.rs
│       │   └── checks.rs       — Sentinel assert, inertia_lut×potentiation >= 1, ...
│       └── bake/               — сборка бинарных блобов
│           ├── mod.rs
│           ├── layout.rs       — SoA упаковка (Columnar, padded_n, warp alignment)
│           ├── sprouting.rs    — compute_power_index, sprouting_score
│           ├── neuron_placement.rs — размещение нейронов в 3D пространстве
│           ├── axon_growth.rs  — рост аксонов, трассировка
│           ├── cone_tracing.rs — конусная трассировка для аксонов
│           ├── dendrite_connect.rs — алгоритм подключения дендритов
│           ├── spatial_grid.rs — пространственная сетка для поиска соседей
│           └── seed.rs         — детерминированный RNG seed
│
├── genesis-runtime/            ← оркестратор + CUDA ядра (Day/Night Cycle)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── main.rs             — точка входа, WebSocket сервер телеметрии
│       ├── config.rs           — конфигурация runtime
│       ├── memory.rs           — управление GPU/CPU памятью
│       ├── ffi.rs              — FFI биндинги к CUDA
│       ├── orchestrator/       — управление Day/Night Cycle, триггеры сна
│       │   ├── mod.rs
│       │   ├── day_phase.rs    — запуск GPU батчей, BSP барьер
│       │   └── night_phase.rs  — Maintenance Pipeline (5 шагов)
│       └── network/            — BSP синхронизация, сетевой стек
│           ├── mod.rs
│           ├── bsp.rs          — Strict BSP, сетевой барьер
│           ├── ring_buffer.rs  — SpikeSchedule, Ping-Pong Double Buffering
│           ├── socket.rs       — TCP/UDP транспорт
│           ├── router.rs       — маршрутизация сообщений
│           ├── slow_path.rs    — медленный путь для служебных сообщений
│           ├── geometry_client.rs — клиент для передачи геометрии нейронов
│           └── telemetry.rs    — WebSocket телеметрия (спайки, состояние)
│
└── genesis-ide/                ← Bevy-based IDE для визуализации и управления
    ├── Cargo.toml
    └── src/
        ├── main.rs             — инициализация Bevy App, регистрация плагинов
        ├── loader.rs           — загрузка геометрии нейронов (WebSocket)
        ├── world.rs            — 3D world rendering, генерация Spike Mesh
        ├── camera.rs           — управление камерой (OrbitCamera)
        ├── hud.rs              — HUD оверлей (Egui): статистика, контролы
        └── telemetry.rs        — приём телеметрии из runtime (спайки, фазы)
```

## Зависимости между крейтами

```
genesis-core  ←─── genesis-baker
genesis-core  ←─── genesis-runtime
genesis-core  ←─── genesis-ide
```

`genesis-baker` и `genesis-runtime` не зависят друг от друга.  
Общий контракт данных (блобы `.state` / `.axons`) — файловый обмен.  
`genesis-ide` подключается к `genesis-runtime` по WebSocket для получения геометрии и телеметрии.

## Коммуникация между компонентами

```
genesis-baker  ──[.state/.axons]──►  genesis-runtime  ──[WebSocket]──►  genesis-ide
```

| Канал | Протокол | Данные |
|---|---|---|
| baker → runtime | Файлы `.baked/` | SoA блобы нейронов и аксонов |
| runtime → ide (геометрия) | WebSocket (JSON) | 3D позиции нейронов |
| runtime → ide (телеметрия) | WebSocket (JSON) | Спайки, Day/Night фаза |

## Соответствие спецификации

| Крейт | Спека |
|---|---|
| `genesis-core` | `07_gpu_runtime.md` §1 (SoA Layout, VariantParameters) |
| `genesis-baker` | `07_gpu_runtime.md` §2.2.4 (Baking), `02_configuration.md`, `04_connectivity.md` §1.6.1 |
| `genesis-runtime` | `05_signal_physics.md`, `06_distributed.md`, `07_gpu_runtime.md` §2 |
| `genesis-ide` | Bevy 0.15, Egui — визуализация и мониторинг |

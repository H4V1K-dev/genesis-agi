# 02. Конфигурация (Configuration)

> Часть архитектуры [Genesis](../../design_specs.md). Файловая структура, пайплайн данных и TOML-спецификации.

---

## 1. Иерархия Конфигурации (Human Readable)

Конфиги делятся на три уровня ответственности. Разделяем, чтобы можно было менять «железо» нейрона, не ломая структуру слоёв.

**Законы Вселенной → Спецификация Зоны → Параметры Инстанса**

### 1.1. Глобальная Физика (`simulation.toml`)

Лежит в корне. Это константы, которые **не могут отличаться** между зонами, иначе сломается синхронизация времени и пространства.

- `tick_duration_us` — Шаг времени (в микросекундах).
- `voxel_size_um` — Размер вокселя (квант пространства).
- `signal_speed_um_tick` — Скорость распространения сигнала.
- `sync_batch_ticks` — Количество тиков автономного расчета между синхронизациями шардов.
- `master_seed` — Глобальный сид (`String`, хэшируется в `u64` при старте) для детерминированной генерации.
- `max_dendrites` — Хард-лимит (128).

**Влияние на память:** Это глобальные `const` переменные (Uniforms в CUDA). Грузятся один раз, всегда в кеше L1.

### 1.2. Конфиг Зоны (Папка `/config/zones/V1/`)
Все файлы конкретной зоны лежат в ее директории. Это позволяет загружать в Runtime несколько зон параллельно из разных папок.
Определяет, **что** мы строим.

1. **`anatomy.toml`** — Слои (L1–L6) в процентах высоты. Позволяет менять масштаб (`world.height`), не переписывая конфиг.
   - Для V1: список слоёв, их высота в %, плотность нейронов.
   - Для «неслоистых» зон: один слой `"Nuclear"` высотой 100%.
   - Распределение типов (4-битная маска: Geo × Sign × Variant) в процентах для каждого слоя.
2. **`blueprints.toml`** — Типы нейронов: параметры мембраны (`threshold`, `leak`) и правила связности (матрица).
   - Параметры GSOP (`74`, `-2`, `64000`).
   - Специфические параметры мембран для каждого из 4-х типов.
3. **`io.toml`** — Карта входов/выходов:
   - Маппинг входных каналов (External Axons → слой L4).
   - Маппинг выходных каналов (аксоны из L5/L6 → Output Ports).
   - Разрешение, стратегия сбора и модальность для каждого канала.

### 1.3. Конфиг Инстанса (`runtime/shard_04.toml` или CLI аргументы)

Определяет, **где** и **какой кусок** считает этот процесс.

- `zone_id` — `"V1"` (ссылка на папку зоны).
- `world_offset` — `{ x: 1000, y: 0, z: 0 }` (смещение этого шарда в глобальном мозге).
- `dimensions` — `{ w: 500, h: 2000 }` (размеры этого куска).
- `neighbors` — Список адресов (`IP:Port` или Shared Memory ID) для соседей (`X+`, `X-`, `Y+`, `Y-`). Поддерживает **Self-loop** (`= Self`) для реализации тороидальной топологии ядерных зон (см. [06_distributed.md §1.5](./06_distributed.md)).

---

## 2. Пайплайн: Запекание (Baking)

**Закон: Движок (Runtime) никогда не читает TOML файлы напрямую.**

Парсинг текстовых конфигов на GPU — смерть производительности. Вводится этап «Запекания» (Baking), который разделяет «Человеческие конфиги» (TOML) и «Машинные данные» (Binary Blobs).

### 2.1. Конвейер

```
TOML конфиги ──→ [Compiler Tool (CPU)] ──→ Бинарные файлы ──→ [Runtime (GPU)]
                  │                          │
                  ├─ Парсит TOML             ├─ .axons (геометрия)
                  ├─ Генерирует нейроны      ├─ .state (начальное состояние)
                  ├─ Растит аксоны           └─ .dendrites (матрица связей)
                  └─ Раскладывает в SoA
```

- **Compiler Tool:** Утилита на CPU — читает TOML, генерирует нейроны, растит аксоны (Cone Tracing), раскладывает данные в плоские массивы.
- **Zero-Copy Loading:** Движок делает `mmap` (отображение файла в память) или прямой `cudaMemcpy` из бинарника в VRAM. Никакого парсинга при старте.

### 2.2. Выравнивание (Alignment)

- Данные на диске лежат **байт-в-байт** так, как они нужны видеокарте.
- Если структура данных занимает 28 байт — добиваем паддингом до **32 байт** для Coalesced Access.
- Время загрузки миллиарда нейронов = скорость чтения SSD.

### 2.3. Валидация Конфигурации (Baking Tool Asserts)

При парсинге `simulation.toml` Baking Tool вычисляет производные величины и проверяет инварианты:

```rust
// baking_compiler/src/physics_constants.rs

let segment_length_um = config.voxel_size_um * config.segment_length_voxels;
let v_seg = config.signal_speed_um_tick / segment_length_um;

// Runtime assert: v_seg обязан быть целым (01_foundations.md §1.6)
assert!(
    config.signal_speed_um_tick % segment_length_um == 0,
    "CRITICAL: signal_speed_um_tick must be a multiple of segment_length_um!"
);
```

### 2.4. Формат `.state`: ShardStateSoA

Бинарный файл `.state` — плоская склейка массивов. Структура **байт-в-байт** совпадает с раскладкой VRAM.

```rust
// baking_compiler/src/layout.rs

/// Warp Alignment: размер кратен 32
pub const fn align_to_warp(count: usize) -> usize {
    (count + 31) & !31
}

pub struct ShardStateSoA {
    pub padded_n: usize,       // Нейроны (aligned to 32)
    pub total_axons: usize,    // Local + Ghost + Virtual (aligned to 32)

    // Soma arrays (size = padded_n)
    pub voltage: Vec<i32>,           // 4 bytes × N
    pub flags: Vec<u8>,              // 1 byte × N (upper nibble: Type, bit 0: Is_Spiking)
    pub threshold_offset: Vec<i32>,  // 4 bytes × N
    pub refractory_counter: Vec<u8>, // 1 byte × N
    pub soma_to_axon: Vec<u32>,      // 4 bytes × N (маппинг Soma → Local Axon ID)

    // Dendrite arrays — Columnar Layout (size = 128 × padded_n)
    // Обращение в CUDA: data[slot * padded_n + tid]
    pub dendrite_targets: Vec<u32>,  // 4 bytes × 128 × N (Packed: 22b Axon_ID | 10b Segment_Index)
    pub dendrite_weights: Vec<i16>,  // 2 bytes × 128 × N (Signed: знак = E/I)
    pub dendrite_timers: Vec<u8>,    // 1 byte × 128 × N

    // Axon arrays (size = total_axons, NOT padded_n!)
    pub axon_heads: Vec<u32>,        // 4 bytes × A (PropagateAxons: += v_seg)
}
```

**Seed-детерминированная генерация:** PackedPosition (`u32`) используется как `entity_id` для хэширования:

```rust
pub fn entity_seed(master_seed: u64, packed_pos: u32) -> u64 {
    wyhash::wyhash(&packed_pos.to_le_bytes(), master_seed)
}
```

### 2.5. Ночной Цикл (Night Baking)

Baking — не одноразовая операция. Каждую «Ночь» (см. [07_gpu_runtime.md](./07_gpu_runtime.md)) Compiler Tool перегенерирует бинарные файлы с учётом изменений (sprouting, nudging, дефрагментация), и движок загружает свежий `.axons` как чистый лист.

---

## 3. Закон Хранения Данных: SoA (Structure of Arrays)

**Полный отказ от объектов в памяти GPU.** Это не рекомендация, это закон архитектуры.

### 3.1. Проблема: AoS (Array of Structures)

```
// Плохо (Cache Miss)
struct Neuron { pos: Vec3, voltage: f32, type_id: u8 }
neurons: [Neuron; N]
```

Чтобы прочитать `voltage` всех нейронов, GPU загружает в кеш `pos` и `type_id` — **мусор**, который не нужен. Полезная нагрузка кеш-линии ≈ 15%.

### 3.2. Решение: SoA

```
// Хорошо (Cache Hit, Coalesced Access)
all_voltages:  [f32; N]    // Непрерывный массив
all_types:     [u8; N]     // Непрерывный массив
all_positions: [Vec3; N]   // Непрерывный массив
```

Варп GPU (32 потока) читает 32 подряд идущих `f32` за **одну транзакцию памяти**. Полезная нагрузка кеш-линии = 100%.

### 3.3. Транспонированные Дендриты (Columnar Layout)

Самое узкое место по памяти: 128 дендритов на нейрон.

- **Не** `Neuron.Dendrites[0..127]` (построчно).
- **А** `Column[Slot_K]` — K-й дендрит для **всех** нейронов подряд (поколонно).

```
Slot_0:   [Нейрон_0.Д0, Нейрон_1.Д0, Нейрон_2.Д0, ...]   // Массив размером N
Slot_1:   [Нейрон_0.Д1, Нейрон_1.Д1, Нейрон_2.Д1, ...]   // Массив размером N
...
Slot_127: [Нейрон_0.Д127, Нейрон_1.Д127, ...]              // Массив размером N
```

В цикле `for slot in 0..128` все потоки GPU обращаются к `Slot_K` — идеально линейное чтение. Bandwidth используется на 100%.

---

## 4. Спецификация: `simulation.toml`

Определяет «песочницу», в которой живет система.

```toml
[world]
# Физические размеры пространства (в микрометрах)
width_um = 3500    # 140 voxels * 25 um
depth_um = 3500    # 140 voxels * 25 um
height_um = 10250  # 410 voxels * 25 um

[simulation]
tick_duration_us = 100  # 1 тик = 100 мкс (0.1 мс). Необходимое разрешение для GSOP.
                        # При velocity=500 мкм/мс сигнал проходит 50 мкм/тик.
total_ticks = 10000     # 10000 тиков = 1 секунда симуляции
master_seed = "GENESIS" # String → хэшируется в u64 при старте

# Глобальная Плотность (Global Density)
# Процент вокселей, в которых находятся тела нейронов (0.04 = 4%).
# Расчет:
# 1. Total_Voxels = (Width_um / 25) * (Depth_um / 25) * (Height_um / 25)
# 2. Total_Neurons = Total_Voxels * global_density
global_density = 0.04
signal_speed_um_tick = 50   # Скорость сигнала (мкм/тик)
segment_length_voxels = 5   # Длина сегмента аксона (вокселей). Глобально. По умолчанию: 5
```

### 4.1. Оптимизация Типов Данных (Memory Optimization)

Понижаем битность счётчиков, опираясь на архитектуру День/Ночь ([07_gpu_runtime.md](./07_gpu_runtime.md)):

| Счётчик | Было | Стало | Обоснование |
|---|---|---|---|
| **Global Clock** | `u64` | `u32` | Хватает на ~5 дней непрерывной работы (при тике 100 мкс). Сбрасывается в фазу Ночи. |
| **Local Timers: Refractory** (soma, synapse) | `u64` | `u8` | Max 255 тиков = 25 мс. Физиологически достаточно для рефрактерности. |
| **Local Timers: Delays** (Delay_Counter) | `u64` | `u16` | 65 535 тиков = 6.5 сек. Межзональные задержки (Projectors). |
| **Batch Size** | `u64` | `u32` | Пакет синхронизации шарда не может быть больше 4 млрд тиков. |

### 4.2. Архитектурные Константы (Immutable)

Параметры, которые «запекаются» при старте и не меняются в рантайме:

| Параметр | Тип | Значение | Примечание |
|---|---|---|---|
| `tick_duration_us` | `u32` | `100` | 0.1 мс |
| `voxel_size_um` | `f32` | `25.0` | Квант пространства |
| `signal_speed_um_tick` | `u16` | `50` | Рассчитывается при старте. `int` для быстрого сложения. |
| `max_dendrites` | `const` | `128` | Hard Constraint. Гарантирует выравнивание памяти по 128 байт (кеш-линии + варпы). |
| `master_seed` | `String` | `"GENESIS"` | Хэшируется в `u64` при старте (см. [01_foundations.md §2](./01_foundations.md)). |

### 4.3. Правило Выравнивания (Warp Alignment)

Требование к Compiler/Baking Tool (§2):

- **Padding Rule:** Количество активных нейронов в бинарном файле шарда (`.state`) всегда добивается пустышками до числа, **кратного 32**.
- **Зачем:** При запуске ядра `UpdateNeurons<<<N/32, 32>>>` все потоки в варпах заняты делом, доступ к памяти строго Coalesced. Без паддинга последний варп может содержать «мусорные» потоки, которые читают за пределы массива.

---

## 5. Спецификация: `anatomy.toml`

Определяет анатомию конкретной зоны: слои, плотность, состав.

### 5.1. Вертикальная Метрика (Relative Height)

- Слои задаются строго в процентах (`height_pct`) от `0.0` до `1.0`.
- **Инвариант:** `L1 + L2 + ... + L6 = 1.0`.
- **Зачем:** Можно менять физическую высоту зоны (растянуть кору с 2 мм до 3 мм) через `world.height`, не переписывая файл анатомии.

### 5.2. Управление Плотностью (Two-Step Density)

Плотность задаётся в два этапа для гибкости:

1. **`global_density`** (в `simulation.toml`): Задаёт «вместимость» всего столба (нейронов на куб. воксель). Рассчитывается `Total_Capacity`.
2. **`population_pct`** (в `anatomy.toml`): Распределяет этот бюджет по слоям.
   - Пример: L4 забирает 35% бюджета (плотный), L1 забирает 1% (почти пустой).

### 5.3. Композиция Типов (Hard Quotas)

> **[!IMPORTANT]**
> Отказ от вероятностной генерации (RNG) в пользу **жёсткого квотирования**.

- **Было (черновик):** «Вероятность спавна типа А = 80%» → при малом N результат плавает.
- **Стало:** «Бюджет типа А = ровно 80% от населения слоя».
  - Если в слое 1000 мест → система **обязана** создать ровно 800 нейронов типа А.
- **Размещение:** Типы перемешиваются в пространстве **детерминированным** алгоритмом (`Shuffle` на базе `master_seed`), чтобы не было кучности. Количество математически точно совпадает с конфигом.

### 5.4. Универсальность Структуры

Схема применима не только к коре:

| Зона | Слоёв | Подход |
|---|---|---|
| **Кора** (V1, V2, Motor) | 6 | `L1..L6` с уникальными `height_pct` и `composition` |
| **Таламус, Базальные ганглии** | 1 | Один слой `"Nuclear"` с `height_pct = 1.0` и своим миксом типов |

Каждая зона (`V1`, `V2`, `Motor`, `Thalamus`) имеет свою отдельную папку и свой `anatomy.toml`.

### 5.5. Пример и Логика Расчёта

```toml
[[layer]]
name = "L4"
height_pct = 0.17
population_pct = 0.35      # Забирает 35% бюджета нейронов

[layer.composition]
# Имя типа (из blueprints.toml) = Жёсткая квота (0.0 - 1.0)
"Vertical_Excitatory" = 0.80   # Ровно 80% населения слоя
"Horizontal_Inhibitory" = 0.20 # Ровно 20% населения слоя

[[layer]]
name = "L2/3"
height_pct = 0.25
population_pct = 0.29

[layer.composition]
"Vertical_Excitatory" = 0.85
"Horizontal_Inhibitory" = 0.15
```

**Логика расчёта при старте (Baking):**

1. `Total_Voxels = (Width_um × Depth_um × Height_um) / voxel_size³`
2. `Total_Capacity = Total_Voxels × global_density`
3. Для каждого слоя:
   - `Z_Start = Σ(предыдущих height_pct) × world.height_um`
   - `Z_End = Z_Start + height_pct × world.height_um`
   - `Layer_Budget = floor(population_pct × Total_Capacity)`
   - Для каждого типа в `composition`:
     - `Type_Count = floor(quota × Layer_Budget)` — **точное число**, не вероятность.
   - Нейроны размещаются в `[Z_Start, Z_End]`, типы перемешиваются `Shuffle(master_seed)`.

---

## 6. Спецификация: `blueprints.toml`

Определяет физические «ТТХ» для каждого из 4-х типов нейронов. Используется **Целочисленная Физика (Integer Physics)** для гарантии детерминизма и снижения нагрузки на FPU. Свойства мембраны не зависят от размера коры — абсолютные величины.

### 6.1. Структура Конфигурации

```toml
[[neuron_type]]
# ID 0 (Биты: 00) — Основной возбуждающий
name = "Vertical_Excitatory"

# --- Параметры Мембраны (Units: microVolts / absolute integers) ---
threshold = 42000               # Порог срабатывания
rest_potential = 10000           # Потенциал покоя
leak_rate = 1200                 # Скорость утечки (вычитание каждый тик)

# --- Тайминги (Units: Ticks) ---
refractory_period = 15           # u8. Абсолютная рефрактерность сомы.
synapse_refractory_period = 15   # u8. Рефрактерность входного порта (дендрита).

# --- Физика Сигнала (Units: Integer geometry) ---
conduction_velocity = 200        # Скорость (дискретное смещение за тик)
signal_propagation_length = 10   # Длина "хвоста" сигнала в сегментах (Active Tail, per-variant)

# --- Гомеостаз (Adaptive Threshold) ---
homeostasis_penalty = 5000       # +5000 к порогу после спайка
homeostasis_decay = 10           # Вычитается из порога каждый тик, пока threshold не вернётся к базовому

# --- Slot Decay (GSOP множители, Fixed-point: 128 = 1.0×) ---
slot_decay_ltm = 160             # Слоты 0-79: 160/128 = 1.25× (усиленное удержание)
slot_decay_wm = 96               # Слоты 80-127: 96/128 = 0.75× (ускоренный распад)

# --- Sprouting Score (веса эвристики выбора аксона, сумма = 1.0) ---
sprouting_weight_distance = 0.5  # f32. Ближний = лучше
sprouting_weight_power   = 0.4  # f32. soma_power_index (См. 04_connectivity.md §1.6.1)
sprouting_weight_explore = 0.1  # f32. Шум по эпохе (защита от повторных выборов)

[[neuron_type]]
# ID 3 (Биты: 11) — Тормозной интернейрон
name = "Horizontal_Inhibitory"

# --- Мембрана ---
threshold = 40000                # Чуть ниже порог (легче возбуждается)
rest_potential = 10000
leak_rate = 1500                 # Быстрее «остывает» (Leak выше)

# --- Тайминги ---
refractory_period = 10           # Быстрее восстанавливается
synapse_refractory_period = 5

# --- Физика Сигнала ---
conduction_velocity = 100        # Медленный сигнал

# --- Гомеостаз ---
homeostasis_penalty = 3000
homeostasis_decay = 15

# --- Slot Decay ---
slot_decay_ltm = 140             # 140/128 = 1.09× (чуть слабее удержание)
slot_decay_wm = 80               # 80/128 = 0.625× (агрессивный распад)

# --- Sprouting Score (интернейроны меньше ценят мощные хабы) ---
sprouting_weight_distance = 0.6  # Больше ценят локальные связи
sprouting_weight_power   = 0.3  # Меньше зависимость от хабов
sprouting_weight_explore = 0.1
```

### 6.2. Типы Данных в Runtime

Пояснения по типам, которые используются в GPU-массивах после Baking (§2):

| Группа | Параметры | Тип Runtime | Обоснование |
|---|---|---|---|
| **Potentials** | `threshold`, `rest_potential`, `voltage` | `i32` | Запас от переполнения при суммации входов |
| **Timers** | `refractory_period`, `synapse_refractory_period` | `u8` | Значения > 255 тиков (25 мс) физиологически бессмысленны для рефрактерности |
| **Geometry** | `conduction_velocity` | `u16` | Дискретные единицы, диапазон до 65535 |
| **Homeostasis** | `homeostasis_penalty`, `homeostasis_decay` | `i32` / `u16` | Penalty суммируется с threshold (i32), decay — малое значение |


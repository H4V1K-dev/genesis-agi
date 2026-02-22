# 07. GPU Runtime (GPU & Storage)

> Часть архитектуры [Genesis](../../design_specs.md). Как данные лежат в VRAM, как грузятся, как переключаются фазы работы.

---

## 1. Архитектура Памяти и Данных (GPU & Storage)

**Инвариант:** Полный отказ от объектов (AoS) в памяти GPU. Данные лежат плоскими векторами (SoA) для обеспечения 100% Coalesced Memory Access (32 потока варпа читают непрерывный кусок памяти за одну транзакцию). Все вычисления — исключительно в целых числах (Integer Physics).

### 1.1. Состояние Сомы (Soma State Layout)

Абстрактные переменные и `f32` **запрещены**. Состояние разбито на плоские выровненные массивы. Размер каждого = `N` (количество нейронов шарда), **выровненное паддингом до числа, кратного 32** (Warp Alignment).

```rust
// Структура массивов в VRAM (SoA), Dense Index 0..N-1
struct SomaState {
    voltage: *mut i32,           // Текущий заряд мембраны (i32 — защита от переполнения)
    threshold_offset: *mut i32,  // Адаптивный порог для гомеостаза (§3.2 из 03_neuron_model)
    refractory_timer: *mut u8,   // Оставшееся время «сна» (0..255 тиков)
    flags: *mut u8,              // Побитовая раскладка ниже
}

// flags (u8) — битовая карта:
// [7..4] Type Mask (Geo | Sign | Variant) — из 03_neuron_model §2
// [3..1] Reserved
// [0]    Is_Spiking (1 = сома выстрелила в этом тике)
```

**Извлечение типа за 1 такт ALU:**

```cuda
u8  f = flags[tid];                       // 32 байта на варп = 1 сектор L1
u8  type_mask = f >> 4;                   // 4-бит тип
u8  sign_bit  = (type_mask >> 1) & 0x1;   // 0 = Excitatory, 1 = Inhibitory
u8  var_id    = (type_mask >> 2) & 0x3;   // Variant → LUT index
```

### 1.2. Транспонированная Матрица Дендритов (Columnar Layout)

Самое узкое место по памяти — 128 дендритных слотов на сому. Чтобы избежать Warp Divergence и кэш-промахов, они хранятся **поколонно**, а не построчно:

```rust
struct DendriteColumns {
    // 128 массивов. В каждом — данные для ВСЕХ нейронов шарда.
    // В цикле `for slot in 0..128` варп читает connected_axon_id[slot]
    // идеально линейно. Bandwidth используется на 100%.
    connected_axon_id: [*mut u32; 128], // Packed: [31..8] Axon_ID (24b), [7..0] Segment_Index (8b)
    synapse_weight:    [*mut i16; 128], // Signed: (+) = Excitatory, (-) = Inhibitory. Знак бейкается.
    refractory_timer:  [*mut u8; 128],  // Локальный таймер невосприимчивости синапса
}
```

### 1.2.1. Аксонные Массивы (Axon State)

Размер = `total_axons` (Local + Ghost + Virtual), **не** `padded_n`. Выровнен до кратного 32.

```rust
struct AxonState {
    head_index:    *mut u32,   // PropagateAxons: += v_seg каждый тик.
    soma_to_axon:  *mut u32,   // Маппинг Dense Soma Index → Axon ID (size = padded_n, Baked)
}
```

**Зонирование `axon_heads` (Baking-time layout):**

```
axon_heads[total_axons]:
┌──────────────────┬──────────────────┬──────────────────┐
│  Local Axons     │  Ghost Axons     │  Virtual Axons   │
│  [0 .. L-1]      │  [L .. L+G-1]    │  [L+G .. A-1]    │
│  soma_to_axon[]  │  ApplySpikeBatch │  InjectInputs    │
└──────────────────┴──────────────────┴──────────────────┘
total_axons = L + G + V (aligned to 32)
```

- **Local:** Родные аксоны сом. Маппинг через `soma_to_axon[dense_id]`.
- **Ghost:** Входящие от соседних шардов. `ApplySpikeBatch` сбрасывает `axon_heads[ghost_id] = 0`.
- **Virtual:** Внешние сенсорные входы. `InjectInputs` сбрасывает `axon_heads[virt_id] = 0`.
- **Инициализация:** Все = `AXON_SENTINEL` (0x80000000). `PropagateAxons` сдвигает **всех** без проверки типа.

### 1.3. Разделение Аксонов: Статика vs Динамика

Паттерны доступа к геометрии и к активности кардинально отличаются — разнесены в разные типы памяти:

| Класс | Содержимое | GPU Memory | Паттерн |
|---|---|---|---|
| **Static Geometry** (Read-Only) | Координаты, длины сегментов, топология графа | Texture Memory / Read-Only Cache | Не изменяется днём. Максимальная скорость чтения. |
| **Dynamic State** (Hot) | `Head_Index` (`u32`) — текущая позиция «поезда» сигнала | Global Memory | Перезаписывается **каждый тик** в горячем цикле. |

### 1.4. Zero-Copy Загрузка (Fast Boot)

Движок в рантайме **не занимается десериализацией** JSON или TOML.

- На этапе Baking CPU формирует бинарные файлы `.state` и `.axons`.
- Их структура **байт-в-байт совпадает** с раскладкой памяти в VRAM (включая padding до 32 байт для выравнивания варпов).
- Загрузка шарда — прямой вызов `cudaMemcpy` (или `mmap`) сырого дампа с NVMe SSD в видеопамять. Время загрузки ограничено только пропускной способностью шины PCIe.

### 1.5. Constant Memory (LUT Layout)

Параметры поведения грузятся в `__constant__` память GPU **один раз** при старте. 256 байт — 100% в L1 кеше, чтение за 1 такт (Broadcast Read при одинаковом Variant в варпе).

```rust
// baking_compiler/src/lut_layout.rs

#[repr(C, align(32))]
pub struct VariantParameters {            // 32 bytes
    pub threshold: i32,                   // 4  — Base threshold
    pub rest_potential: i32,              // 4  — Rest potential (GLIF reset)
    pub leak: i32,                        // 4  — GLIF Leakage per tick
    pub homeostasis_penalty: i32,         // 4  — Penalty on spike
    pub homeostasis_decay: i32,           // 4  — Decay per tick (i32: zero casts)
    pub gsop_potentiation: u16,           // 2  — Unsigned delta (sign in weight)
    pub gsop_depression: u16,             // 2  — Unsigned delta
    pub refractory_period: u8,            // 1  — Soma refractory (ticks)
    pub synapse_refractory: u8,           // 1  — Synapse refractory (ticks)
    pub slot_decay_ltm: u8,               // 1  — Множитель GSOP для LTM (слоты 0-79). Fixed-point: 128 = 1.0×
    pub slot_decay_wm: u8,                // 1  — Множитель GSOP для WM (слоты 80-127). Fixed-point: 128 = 1.0×
    pub _padding: [u8; 4],               // 4  → Total = 32 bytes
}

#[repr(C, align(128))]
pub struct GenesisConstantMemory {        // 256 bytes
    pub variants: [VariantParameters; 4], // 128 bytes (4 × 32)
    pub inertia_lut: [u8; 16],            // 16 bytes (16 рангов: abs(weight) >> 12)
    pub _padding: [u8; 112],              // → Total = 256 bytes
}
```

**Доступ из CUDA kernel:** Variant ID извлекается за 1 такт `(flags[tid] >> 6) & 0x3`, далее прямое чтение: `const_mem.variants[variant].threshold`.

> **⚠️ Baking Tool Validator (inertia_lut):** При сборке `GenesisConstantMemory` необходимо проверять, что минимальный результат `(gsop_potentiation * inertia_lut[rank]) >> 7 >= 1` для **всех** рангов и вариантов. Если результат равен 0, возникает «Мёртвая зона пластичности» — вес синапса перестаёт адаптироваться навсегда. Это задача валидатора Baking Tool (и в перспективе IDE с live-подсказками по конфигу).

---

## 2. Архитектура Цикла: День и Ночь (Day/Night Cycle)

Фундаментальное решение: разделение вычислений во времени. Разрешает конфликт между жёсткой статической памятью (Coalesced Access на GPU) и структурной пластичностью графа (динамические аллокации).

**Инвариант:** Night Phase — **локальная операция на уровне зоны**. Замораживается только конкретная зона — остальные продолжают работать. Глобального останова нет.

### 2.1. Фаза «День» (Online / Hot Loop)

Выполняется **исключительно на GPU**. Максимальная пропускная способность, полностью лишена структурной логики.

- **Read-Only Топология:** Геометрия аксонов и массив подписок дендритов заморожены. Никаких `malloc`/`free` внутри ядра.
- **Изменяемое Состояние:** Веса синапсов (GSOP), `axon_heads[]`, таймеры, вольтаж, флаги.

**Порядок запуска ядер (каждый тик):**

| # | Kernel | Описание |
|---|---|---|
| 1 | `InjectInputs` | Bitmask Injection для виртуальных аксонов (single-tick pulse) |
| 2 | `ApplySpikeBatch` | Сброс `axon_heads[ghost_id] = 0` для входящих сетевых спайков |
| 3 | `PropagateAxons` | Безусловный `axon_heads[tid] += v_seg` для **всех** аксонов (Local + Ghost + Virtual) |
| 4 | `UpdateNeurons` | GLIF: leak → dendrite loop → threshold → fire. При спайке: `axon_heads[my_axon] = 0` |
| 5 | `ApplyGSOP` | Пластичность: Timer-as-Contact-Flag (`timer == synapse_refractory` → Potentiation, иначе Depression) |
| 6 | `RecordOutputs` | Z-Sort map → `u8` per channel, Coalesced Write в `output_history[]` |


### 2.2. Фаза «Ночь» (Per-Zone Offline Maintenance)

Выполняется на **CPU**. Каждая зона имеет **свой цикл сна** — независимый от остальных.

**Триггеры засыпания:**
Проверяются оркестратором на CPU **только в конце каждого батча** `sync_batch_ticks` (во время сетевого барьера). Внутри рантайма GPU проверок сна нет — такты не тратятся.

| Триггер | Источник | Пример |
|---|---|---|
| **Таймер** | `night_interval_ticks` в конфиге зоны | V1: каждые 5 мин, гиппокамп: каждые 2 мин |
| **Внешний сигнал** | `sleep_zone(zone_id)` через API оркестратора | Массовый сон существа (моторика → сенсоры → ассоциация) |
| **Никогда** | `night_interval_ticks = 0` | Статические зоны (таламус, ствол — Variant = Fixed/Relay, GSOP заморожен) |

> **⚠️ Sentinel Refresh (зоны с `night_interval_ticks = 0`):**
> `AXON_SENTINEL = 0x80000000` ≈ 59.6 часов при v_seg=1. Без Night Phase неактивные аксоны переполнятся → фантомные спайки. **Решение:** Каждые ~50 часов (`SENTINEL_REFRESH_TICKS = 1_800_000_000`) host запускает лёгкий проход: все `axon_heads[id]` со значением `> SENTINEL_DANGER_THRESHOLD` принудительно сбрасываются в `AXON_SENTINEL`. Активные сигналы (head < propagation_length × 10) не затрагиваются.

**Конвейер Maintenance (5 шагов):**

| Шаг | Где | Название | Описание |
|---|---|---|---|
| **1** | **GPU** | **Sort & Prune** | Segmented Radix Sort: 128 слотов по `abs(weight)` (descending). Слоты с `abs(w) < threshold` обнуляются. Шина PCIe не забивается мусором. |
| **2** | **PCIe** | **Download** (VRAM → RAM) | `cudaMemcpyAsync` только изменённых массивов (веса + targets). Статическая геометрия уже известна хосту. |
| **3** | **CPU** | **Sprouting & Nudging** | Тяжёлая фаза. Cone Tracing для пустых слотов (Spatial Hash), рост отростков, создание Ghost Axons для межшардовых путей. Длительность зависит от железа и turnover rate. |
| **4** | **CPU** | **Baking** | Дефрагментация топологии → новый `.axons`. Подготовка SoA-массивов с выравниванием по 32 (Warp Alignment). |
| **5** | **PCIe** | **Upload** (RAM → VRAM) | `cudaMemcpyAsync` свежих данных. Шард мгновенно встраивается в барьер Strict BSP. |

**Длительность фазы Maintenance — плавающая.** Зависит от количества нейронов, turnover rate и мощности CPU. Быстрый CPU = быстрый метаболизм, короткий сон. Это легализовано через [Structural Determinism](./01_foundations.md) (§2.3).

**Возвращение по готовности:** Как только `cudaMemcpy` завершается, шард мгновенно встраивается в текущий барьер Strict BSP и продолжает работу.

#### 2.2.1. Step 1: GPU Sort & Prune (Детали)

**Проблема:** 128 дендритов лежат поколонно (Columnar Layout, stride = N). Глобальный Radix Sort с таким stride убьёт кэш.

**Решение: Shared Memory Staging**

1. Ядро загружает 128 слотов для 32 нейронов варпа в Shared Memory (AoS [Neuron][Slot])
   - Per slot: `weight` (i16, 2B) + `target` (u32, 4B) + `timer` (u8, 1B) = **7 bytes**
   - Per neuron: 128 × 7 = **896 bytes**
   - Per warp (32 neurons): **~28 KB** → идеально в Shared Memory (48-96 KB/SM)

2. **Bitonic Sort** (лучше Radix для N=128 на GPU) по `abs(weight)` descending — целочисленный, без float

3. **Auto LTM/WM Promotion:** Сортировка автоматически ставит сильнейшие связи в слоты 0-79 (LTM, low decay), слабые — в 80-127 (WM, high decay). Никакой ручной логики перемещения.

4. **Pruning:** Слоты в хвосте с `abs(weight) < prune_threshold` → `target_packed = 0` (Sentinel пустого слота)

5. Запись обратно в глобальную память в Columnar Layout

```
Shared Memory (AoS, per warp):
┌─────────────────────────────────────────────────┐
│ Neuron 0: [slot0_w, slot0_t, slot0_tmr] ... ×128│
│ Neuron 1: [slot0_w, slot0_t, slot0_tmr] ... ×128│
│ ...                                      ×32    │
└───────── Sort per neuron, write back ───────────┘
```

> **`weight = 0` ≠ `target = 0`:** Днём вес может упасть до 0 через GSOP depression — связь электрически молчит, но структурно жива (target ≠ 0). GSOP может поднять её обратно. Физическое удаление (`target = 0`) — только здесь, при Pruning.

#### 2.2.3. Step 3: Sprouting & Nudging (CPU, f32 легален)

Порядок строго последовательный: сначала растим кабели, потом ищем розетки.

**a) Nudging (Growth Step):**
- Аксоны с `remaining_length > 0` делают шаг через `step_and_pack()` (см. [04_connectivity.md §4.3](./04_connectivity.md)).
- Математика: `V_global + V_attract + V_noise` → `normalize` → `quantize` → `PackedPosition`.

**b) Boundary Check → NewAxon Handover:**
- Если координата вылетает за габариты шарда → аксон обрезается, формируется `NewAxon { entry_point, vector, type_mask }` в Slow Path очередь соседу (см. [06_distributed.md §2.5](./06_distributed.md)).

**c) Spatial Grid Rebuild:**
- Новые сегменты прописываются в 3D хэш-сетку (ключи из PackedPosition X|Y|Z). Обязателен до Sprouting — иначе `get_in_radius()` не увидит свежие аксоны.

**d) Sprouting (Slot Filling):**
- CPU сканирует массив `targets[]`. Если `target_packed == 0` — слот пуст.
- Cone Query: `calculate_v_attract()` в Spatial Grid (FOV + Lookahead).
- Фильтрация: тип владельца аксона = `seg_val >> 28` (4 бита из PackedPosition). Без обращения к соме.
- **Выбор кандидата** — тройной скоринг `sprouting_score()` по дистанции, `soma_power_index` и exploratory-шуму (см. [04_connectivity.md §1.6.1](./04_connectivity.md)). Веса конфигурируются по типу нейрона.
- Новый `target_packed` записывается, вес = базовый (74), слот попадает в WM (индексы 80-127).

#### 2.2.4. Step 4: Baking & Defragmentation (CPU)

**a) f32 → u32 Quantization:**
- Float-координаты квантуются через `step_and_pack()` → `PackedPosition` (4 bytes/segment).

**b) DenseIndex Generation:**
- GPU работает с dense indices (0..N-1), не с PackedPosition.
- CPU строит маппинг: `PackedPosition → dense_id` для всех `target_packed` в массиве дендритов.
- В массив `targets[]` вписываются DenseIndex + segment offset.

**c) Columnar Layout Defrag:**
- Новые связи вписываются в транспонированную матрицу `Column[Slot_K]`, не в конец массива.

**d) Warp Alignment:**
- `padded_n = align_to_warp(neuron_count)` → padding до кратного 32.
- Итоговые `.state` и `.axons` блобы байт-в-байт совпадают с VRAM layout → Step 5: `cudaMemcpyAsync`.

### 2.3. Легализованная Амнезия (Spike Drop)

Пока зона спит, остальные зоны продолжают работать и слать спайки (Fast Path).

- Хост спящей зоны принимает TCP/UDP пакет, видит статус `SLEEP` → **мгновенный Drop**.
- Ноль копирований в VRAM. Ноль ветвлений. Информация теряется **физиологически достоверно**.
- **Биологический аналог:** Человек во сне не обрабатывает зрительный вход. Это нормальное поведение живой системы, не ошибка инференс-сервера.

---

## 3. Инварианты Жизненного Цикла (Lifecycle Invariants)

### 3.1. Cold Start: Sentinel Assert

> **⚠️ Baking Tool Assert:** Перед записью `.state` блоба Baking Tool обязан убедиться что весь массив `axon_heads` заполнен `AXON_SENTINEL` (`0x80000000`), а не нулями (`calloc`-default). Нули при старте вызовут эпилептический разряд всей коры в Тик 1 — гомеостатические пороги задерутся и система умрёт на старте.

```rust
// baking_compiler/src/validate.rs
assert!(
    axon_heads.iter().all(|&h| h == AXON_SENTINEL),
    "CRITICAL: axon_heads must be initialized to AXON_SENTINEL, not zero!"
);
```

### 3.2. Reset: O(1) Сброс и Блокирующая Ночь

При команде `reset_zone(zone_id)`:

1. **Если зона спит (Night Phase):** Сброс **блокирующий** — CPU дожидается завершения Maintenance pipeline (до Step 5 Upload включительно). Прерывание в середине оставит VRAM с дырявыми матрицами дендритов.
2. **Ring Buffer инвалидация (O(1)):** Обнуляются только `counts` обоих Ping-Pong буферов. Сами `ghost_id` не важны — GPU читает ровно `counts[tick]` записей. Предотвращает фантомные сигналы из прошлой жизни.

```rust
// O(1) — достаточно обнулить счётчики, не весь буфер
memset(schedule_a.counts, 0, batch_size * size_of::<u32>());
memset(schedule_b.counts, 0, batch_size * size_of::<u32>());
```

> **Phantom Signals & Input Bleed:** Фантомные сигналы из Ring Buffer при перезапуске — **легализованное биологическое поведение** (аналог дежавю при пробуждении). Input Bleed от асинхронного сенсора — аналогично. Не дефекты архитектуры.

### 3.3. Hot Checkpoint (Периодический Дамп на Диск)

Помимо дампа геометрии после каждой Night Phase, оркестратор делает **периодический снапшот** (`dendrite_weights` + `dendrite_targets`) в холодное хранилище:

```rust
const CHECKPOINT_INTERVAL_BATCHES: u32 =
    300_000_000 / TICK_DURATION_US / SYNC_BATCH_TICKS; // ≈ 5 минут

if batch_counter % CHECKPOINT_INTERVAL_BATCHES == 0 {
    cudaMemcpyAsync(host_buf, vram_weights, ..., DeviceToHost);
    // Атомарная запись: сначала .tmp, потом rename() — защита от краша
    write_to_disk("checkpoint_weights.bin.tmp");
    rename("checkpoint_weights.bin.tmp", "checkpoint_weights.bin");
}
```

| Тип дампа | Триггер | Файл |
|---|---|---|
| **Геометрия** (`axons`) | После каждой Night Phase | `.axons` |
| **Состояние** (`weights` + `targets`) | Каждые ~5 минут | `checkpoint_weights.bin` |



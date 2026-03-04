# 07. GPU Runtime (GPU & Storage)

> Часть архитектуры [Genesis](../../design_specs.md). Как данные лежат в VRAM, как грузятся, как переключаются фазы работы.

---

## 1. Архитектура Памяти и Данных (GPU & Storage)

**Инвариант:** Полный отказ от объектов (AoS) в памяти GPU. Данные лежат плоскими векторами (SoA) для обеспечения 100% Coalesced Memory Access (32 потока варпа читают непрерывный кусок памяти за одну транзакцию). Все вычисления — исключительно в целых числах (Integer Physics).

### 1.1. Строгий FFI-Контракт VRAM (Headerless SoA)

Движок не парсит данные. Бинарные файлы `.state` и `.axons` — это чистые дампы памяти (Headerless), готовые к `cudaMemcpyAsync`.

**Размер полезной нагрузки строго детерминирован: 782 байта на нейрон** (с учётом 128 слотов дендритов).

**Раскладка `ShardVramPtrs` (порядок байт в .state блобе):**

1. **`soma_voltage`**: 4 байта (i32) — текущий заряд мембраны
2. **`soma_flags`**: 1 байт (u8) — включает 4-битный Type ID
   - `[7..4]`: Type Mask (Geo | Sign | Variant)
   - `[3..1]`: Reserved
   - `[0]`: Is_Spiking (1 = выстрелила в этом тике)
3. **`threshold_offset`**: 4 байта (i32) — адаптивный порог гомеостаза
4. **`timers`**: 1 байт (u8) — таймеры рефрактерности сомы и синапсов
5. **`soma_to_axon`**: 4 байта (u32) — маппинг soma_id → axon_id
6. **`dendrite_targets`**: 512 байт (u32 × 128) — Columnar Layout
   - Каждый slot: `[31..8]` Axon_ID (24b) | `[7..0]` Segment_Index (8b)
7. **`dendrite_weights`**: 256 байт (i16 × 128) — Columnar Layout
   - Signed: `(+)` = Excitatory, `(-)` = Inhibitory. Знак закрыт при Baking.

**Итого: 4 + 1 + 4 + 1 + 4 + 512 + 256 = 782 байта/нейрон**

**Инвариант Выравнивания:** Все массивы имеют длину `padded_n`, кратную 32 (Warp Alignment). Дамп памяти хранится в Little-Endian, без заголовков, без метаданных. Baker гарантирует byte-for-byte совпадение дампа с VRAM layout.

**Извлечение типа за 1 такт ALU:**

```cuda
u8  f = flags[tid];                       // 1 байт, 32 байта на варп = 1 сектор L1
u8  type_mask = f >> 4;                   // 4-бит тип
u8  sign_bit  = (type_mask >> 1) & 0x1;   // 0 = Excitatory, 1 = Inhibitory
u8  var_id    = (type_mask >> 2) & 0x3;   // Variant → LUT index в VariantParameters
```

### 1.2. Константная Память (VariantParameters)

Параметры мембраны и пластичности загружаются в `__constant__` память GPU один раз при старте.

**Размер структуры строго 64 байта для идеального попадания в L1 Cache.**

```cpp
struct VariantParameters {
    int32_t threshold;                      // 4 б. — Base threshold
    int32_t rest_potential;                 // 4 б. — Rest potential (GLIF reset)
    int32_t leak;                           // 4 б. — GLIF Leakage per tick
    int32_t homeostasis_penalty;            // 4 б. — Penalty on spike
    int32_t homeostasis_decay;              // 4 б. — Decay per tick
    uint16_t gsop_potentiation;             // 2 б. — Unsigned delta (sign in weight)
    uint16_t gsop_depression;               // 2 б. — Unsigned delta
    uint8_t refractory_period;              // 1 б. — Soma refractory (ticks)
    uint8_t synapse_refractory_period;      // 1 б. — Synapse refractory (ticks)
    uint8_t slot_decay_ltm;                 // 1 б. — Множитель GSOP для LTM (0-79). Fixed: 128 = 1.0×
    uint8_t slot_decay_wm;                  // 1 б. — Множитель GSOP для WM (80-127). Fixed: 128 = 1.0×
    uint8_t signal_propagation_length;      // 1 б. — Active Tail length
    uint8_t ltm_slot_count;                 // 1 б. — LTM vs WM threshold boundary
    uint8_t inertia_curve[16];              // 16 б. — Inertia modifiers (rank: abs(weight) >> 11)
    uint8_t _padding[14];                   // 14 б. → Total = 64 bytes
};

// Контейнер для 16 вариантов
struct GenesisConstantMemory {
    VariantParameters variants[16];          // 1024 bytes total
};   // Умещается в константную память GPU (64KB limit)
```

**Доступ из CUDA kernel:** Variant ID извлекается из `soma_flags` за 1 такт: `var_id = (flags[tid] >> 6) & 0x3`. Далее — прямое чтение из L1 (Broadcast Read за 1 такт):

```cuda
int32_t threshold = const_mem.variants[var_id].threshold;
int32_t rest_potential = const_mem.variants[var_id].rest_potential;
// и т.д. — все параметры доступны за 1 такт ALU на весь варп
```

**Инвариант:** Ни один разработчик не должен добавлять поля после `_padding`. Это жёсткий контракт VRAM. Любое расширение требует версионирования структуры и переподготовки Baking Tool.

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
| **Static Geometry** (Read-Only) | Координаты, длины сегментов, топология графа (`soma_to_axon`, `dendrite_targets`) | L1 Read-Only Data Cache | Не изменяется днём. Чтение идёт через `const __restrict__` указатели в ядрах. Максимальная скорость. |
| **Dynamic State** (Hot) | Вектора сигналов (`axon_heads`), веса синапсов (`dendrite_weights`), таймеры, вольтаж, флаги | Global Memory (L1 RW Cache) | Перезаписывается **каждый тик** в горячем цикле. |

### 1.4. Zero-Copy Загрузка (Fast Boot)

Движок в рантайме **не занимается десериализацией** JSON или TOML.

- На этапе Baking CPU формирует бинарные файлы `.state` и `.axons`.
- Их структура **байт-в-байт совпадает** с раскладкой памяти в VRAM (включая padding до 32 байт для выравнивания варпов).
- Загрузка шарда — прямой вызов `cudaMemcpy` (или `mmap`) сырого дампа с NVMe SSD в видеопамять. Время загрузки ограничено только пропускной способностью шины PCIe.

### 1.5. Constant Memory (LUT Layout)

Параметры поведения грузятся в `__constant__` память GPU **один раз** при старте. Структура `GenesisConstantMemory` занимает 1024 байта (16 вариантов по 64 байта), что идеально помещается в 64KB лимит и обеспечивает Broadcast Read за 1 такт, если все треды в варпе имеют одинаковый Variant.

```rust
#[repr(C, align(64))]
pub struct VariantParameters {            // 64 bytes
    pub threshold: i32,                   // 4  — Base threshold
    pub rest_potential: i32,              // 4  — Rest potential (GLIF reset)
    pub leak: i32,                        // 4  — GLIF Leakage per tick
    pub homeostasis_penalty: i32,         // 4  — Penalty on spike
    pub homeostasis_decay: i32,           // 4  — Decay per tick (i32: zero casts)
    pub gsop_potentiation: u16,           // 2  — Unsigned delta (sign in weight)
    pub gsop_depression: u16,             // 2  — Unsigned delta
    pub refractory_period: u8,            // 1  — Soma refractory (ticks)
    pub synapse_refractory: u8,           // 1  — Synapse refractory (ticks)
    pub slot_decay_ltm: u8,               // 1  — Множитель GSOP для LTM (0-79). Fixed: 128 = 1.0×
    pub slot_decay_wm: u8,                // 1  — Множитель GSOP для WM (80-127). Fixed: 128 = 1.0×
    pub propagation_length: u8,           // 1  — Active Tail length
    pub ltm_slot_count: u8,               // 1  — LTM vs WM threshold boundary
    pub inertia_curve: [u8; 16],          // 16 — Inertia modifiers (rank: abs(weight) >> 11)
    pub _padding: [u8; 14],               // 14 → Total = 64 bytes
}

#[repr(C, align(128))]
pub struct GenesisConstantMemory {        // 1024 bytes
    pub variants: [VariantParameters; 16], // 16 variants supported
}
```

**Доступ из CUDA kernel:** Variant ID извлекается из `flags` за 1 такт: `(flags[tid] >> 4) & 0xF`. Далее прямое чтение: `const_mem.variants[variant].threshold`.

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

| # | Kernel | Описание | Оисточник |
|---|---|----|----|
| 1 | `InjectInputs` | Bitmask Injection для виртуальных аксонов (single-tick pulse) | [05_signal_physics.md §2.4](./05_signal_physics.md) |
| 2 | `ApplySpikeBatch` | Чтение Ghost indices из Schedule, сброс `axon_heads[ghost_id] = 0` | [05_signal_physics.md §1.2.1](./05_signal_physics.md) |
| 3 | `PropagateAxons` | Безусловный `axon_heads[tid] += v_seg` для **всех** аксонов (Local + Ghost + Virtual) | [05_signal_physics.md §1.6](./05_signal_physics.md) |
| 4 | `UpdateNeurons` | GLIF + дендритный цикл + проверка порога + срыв спайка | [05_signal_physics.md §1.5](./05_signal_physics.md) |
| 5 | `ApplyGSOP` | Пластичность: Timer-as-Contact-Flag режим STDP | [05_signal_physics.md §1.3](./05_signal_physics.md) |
| 6 | `RecordReadout` | Чтение spike flags из mapped_soma_ids, запись в output_history | [05_signal_physics.md §3.2](./05_signal_physics.md) |


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

### 2.3. External I/O Server (UDP для входов/выходов)

Отдельный Tokio-сервер (на третьем ядре) для взаимодействия с External Hub. Обрабатывает I/O неблокирующе.

```rust
pub struct ExternalIoServer {
    sock_in: Arc<UdpSocket>,        // Port N: ресивер Input Bitmasks
    sock_out: Arc<UdpSocket>,       // Port N+1: сендер Output History
    last_client_addr: Option<SocketAddr>, // Память о клиенте
}

// Протокол пакета
#[repr(C)]
pub struct ExternalIoHeader {
    pub zone_hash: u32,     // идентификатор Zone
    pub matrix_hash: u32,   // идентификатор Input/Output матрицы
    pub payload_size: u32,  // размер пайлоада
}
```

**Дисфрагментация:** UDP пакеты больше 65KB автоматически дропятся (отсутствует EMSGSIZE отравления сокета). Полные передачи когда батч готов.

**Плагин**:
- На каждом батче (когда `current_tick_in_batch == 0`) стадия услуги выслыла UDP датаграмму с `output_history` предыдущего батча клиенту (робоцика, визуализация).
- Одновременно вычитывает входящие `Input Bitmask` из датаграмм, сканирует через `try_recv_input()` в неблокирующем положении и ассоциирует пиксели с Virtual Axons (`InjectInputs`).

---

## Связанные документы

| Документ | Что связывается |
|---|---|
| [05_signal_physics.md](./05_signal_physics.md) | Day Pipeline kernels (§1.0), Constant Memory параметры |
| [06_distributed.md](./06_distributed.md) | Ring Buffer, Ghost Axons, BSP sync, сетевой I/O |
| [02_configuration.md](./02_configuration.md) | Определения Variant'ов, blueprints, валидация параметров |
| [09_baking_pipeline.md](./09_baking_pipeline.md) | .state/.axons формат файле, Sort&Prune в Night |
| [project_structure.md](../project_structure.md) | Обзор архитектуры |

---

## Changelog

| Дата | Версия | Описание |
|---|---|---|
| 2026-02-28 | 2.1 | Синхронизирована VramState с реальным memory.rs (добавлены I/O Matrix поля, readout буферы). Обновлена таблица Day Phase с 6 kernels и ссылками на источники. Добавлен раздел External I/O Server для UDP мультиплексирования. |
| TBD | 2.0 | Первая версия |
- CPU строит маппинг: `PackedPosition → dense_id` для всех `target_packed` в массиве дендритов.
- В массив `targets[]` вписываются DenseIndex + segment offset.

**c) Columnar Layout Defrag:**
- Новые связи вписываются в транспонированную матрицу `Column[Slot_K]`, не в конец массива.

**d) Warp Alignment:**
- `padded_n = align_to_warp(neuron_count)` → padding до кратного 32.
- Итоговые `.state` и `.axons` блобы байт-в-байт совпадают с VRAM layout → Step 5: `cudaMemcpyAsync`.

### 2.3. External I/O Server (UDP для вѝԳется/выходов)

Отдельный Tokio-сервер (третью ѰЯдро) для вчёт-раза Internal Compute. Обрабатывает I/O неавтонно.

```rust
pub struct ExternalIoServer {
    sock_in: Arc<UdpSocket>,        // Port N: ресивер Input Bitmasks
    sock_out: Arc<UdpSocket>,       // Port N+1: сендер Output History
    last_client_addr: Option<SocketAddr>, // Мемория о клиенте
}

// Протокол пакета
#[repr(C)]
pub struct ExternalIoHeader {
    pub zone_hash: u32,     // идентификатор Zone
    pub matrix_hash: u32,   // идентификатор Input/Output матрицы
    pub payload_size: u32,  // размер пайлоада
}
```

**ДислокČрагментирование:** UDP пакеты больше 65KB автоматически дропъются (отсутствует EMSGSIZE потравления сокета). Полные алвания когда батч готов.

**ПԼтАгтѯн**:
- На каждом батче (когда `current_tick_in_batch == 0`) нчамск услюгные нервые высылают UDP датаграммю с `output_history` предыдущего батча клиенту (роботика, визуализация).
- Одновременно вычитывает входящие `Input Bitmask` из датаграмм, сканирует через `try_recv_input()` в неблокирующем положенны и ассоциирует пиксели с Virtual Axons (`InjectInputs`).

### 2.4. Легализованная Амнезия (Spike Drop)

Пока зона спит, остальные зоны продолжают работать и слать спайки (Fast Path).

- Хост спящей зоны принимает TCP/UDP пакет, видит статус `SLEEP` → **мгновенный Drop**.
- Ноль копирований в VRAM. Ноль ветвлений. Информация теряется **физиологически достоверно**.
- **Биологический аналог:** Человек во сне не обрабатывает зрительный вход. Это нормальное поведение живой системы, не ошибка инференс-сервера.

---

## 3. Инварианты Жизненного Цикла (Lifecycle Invariants)

---

## Connected Documents

| Document | Connection |
|---|---|
| [05_signal_physics.md](./05_signal_physics.md) | Day Pipeline kernels (§1.0), Constant Memory variant parameters |
| [06_distributed.md](./06_distributed.md) | Ring Buffer, Ghost Axons, BSP sync, network I/O |
| [02_configuration.md](./02_configuration.md) | Variant definitions, blueprints, parameter validation |
| [09_baking_pipeline.md](./09_baking_pipeline.md) | .state/.axons file format, Sort&Prune during Night |
| [project_structure.md](../project_structure.md) | Architecture overview |

---

## Changelog

| Date | Version | Changes |
|---|---|---|
| 2026-02-28 | 2.1 | Synchronized VramState with real memory.rs (added I/O Matrix fields, readout buffers). Updated Day Phase table with 6 kernels and source references. Added External I/O Server section for UDP multiplexing. |
| TBD | 2.0 | First version |

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



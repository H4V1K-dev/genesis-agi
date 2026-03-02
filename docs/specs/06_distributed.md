# 06. Распределённая Архитектура (Distributed)

> Часть архитектуры [Genesis](../../design_specs.md). Шардинг, BSP, границы, атлас-маршрутизация.

**Статус:** [MVP] = реализовано в V1, [PLANNED] = дорабатывается после V1.0 release.

---

## 1. Планарный Шардинг (Tile-Based Sharding)

### 1.1. Обоснование [MVP]

1. **Топология коры:** Кора — это «простыня» (Sheet). 2–4 мм толщины (Z) против метров ширины (X/Y). Пилить 4 мм на разные сервера — безумие. Пилить метры ширины на тайлы — необходимость.
2. **Колоночная архитектура:** Вся оптимизация памяти (Columnar Layout, см. [02_configuration.md §3](./02_configuration.md)) заточена на то, что вертикальная колонка (L1–L6) лежит в памяти рядом. Внутри колонки самый горячий трафик (L4 → L2/3 → L5 → L6). Разрезать Z = превратить локальные связи в сетевые пакеты.
3. **Атлас вместо стэка:** Для соединения V1 (затылок) с PFC (лоб) используем маршрутизацию через Атлас (§3), а не физическое вертикальное наложение.

### 1.2. Топология Соседей [MVP]

Каждый шард — плоская плитка (Tile). Максимум **4 соседа:**

```
        [North (Y+)]
            │
[West (X-)]─┼─[East (X+)]
            │
        [South (Y-)]
```

Вертикальных соседей нет. Все связи по Z (L2 → L5, L4 → L6) — **всегда локальные** (Zero-Copy).

### 1.3. Ghost Connections (Ghost Axon Metadata) [MVP]

При прорастании аксона через боковую границу шарда формируется контакт (Connection). В текущей MVP реализации (V1) сохраняется только метаинформация о связи:

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GhostConnection {
    pub local_axon_id: u32,    // Индекс аксона на целевом шарде (Ghost)
    pub paired_src_soma: u32,  // Soma ID на источнике (для трассировки)
}
```

**Жизненный цикл:**
- **Baking Phase:** Baker создаёт файл `.ghosts` с массивом Connections, отсортированными по целевому каналу.
- **Runtime:** Каждая Connection описывает Ghost Axon, который уже **физически растёт** в целевом шарде (как обычный локальный аксон).
- **Sender-Side Mapping:** При спайке soma на источнике, спайк кодируется как предвычисленный `ghost_id` (из таблицы маппинга), передаётся по сети, и на целевом шарде используется прямым индексом в массив `axon_heads`.

**[PLANNED] Полная AxonHandover (V2):**
При включении полнофункционального роста аксонов через границы, будет расширено до:

```rust
struct AxonHandover {
    coord_a: u16,              // Координата на стыке (Y при X-граница | X при Y-граница)
    coord_b: u16,              // Z-координата

    dir_x: i8,                 // Вектор роста (нормализованный)
    dir_y: i8,
    dir_z: i8,

    type_mask: u8,             // Geo(1) | Sign(1) | Var(2)
    remaining_segments: u16,   // TTL роста
    _pad: u8,
}
```

**Отличие:** In V1, аксон уже вырос и зафиксирован. In V2, аксон продолжал бы расти через границы динамически.

### 1.4. Ghost Axons (Планарная Версия) [PLANNED]

Механика из [04_connectivity.md §1.7](./04_connectivity.md) с поправкой на 2D-границы:

- **Рождение:** При получении `AxonHandover` шард создаёт Ghost Axon от стены внутрь.
- **Цепочка:** Если Ghost дорастает до **противоположной** стены (X- → X+), экспортируется дальше.
- **Вертикальная свобода:** Ghost Axon может расти вверх/вниз по Z свободно — это локальная память, без сетевого трафика.

### 1.5. Тороидальная Топология (Periodic Boundaries) [PLANNED]

Для ядерных неслоистых структур (таламус, базальные ганглии) применяются **Периодические Граничные Условия**. Физика шара через математику плоскости.

**Принцип:** GPU по-прежнему считает плоскую сетку. Замыкание решается **конфигурацией соседей:**

- **1 шард (весь объём на одной GPU):** `Neighbor_X+ = Self`, `Neighbor_X- = Self`, `Neighbor_Y+ = Self`, `Neighbor_Y- = Self`. Аксон, вышедший за правый край, появляется на левом как Ghost Axon **того же шарда**.
- **Сетка шардов (например, 2×2):** Правый крайний шард → `Neighbor_X+ = левый крайний шард` (кольцевое замыкание).

**Ось Z:** Для слоистой коры Z жёстко ограничен (L1 сверху, L6 снизу) — замыкание **запрещено**. Для ядерных зон (где `height_pct = 1.0` и нет слоёв) — Z замыкается аналогично X/Y.

**Защита от Ouroboros (бесконечных циклов):** Поле `remaining_segments` в `AxonHandover` (§1.3) продолжает уменьшаться при каждом пересечении границы. Когда `remaining_segments == 0` — рост останавливается. Аксон физически не может замкнуться на себя, если его длина меньше периметра зоны.

**Итог:** Нет «мёртвых зон» на краях. Изотропия сохранена. Ноль нового кода в Hot Loop.

---

## 2. Протокол Синхронизации (Time Protocol) [MVP]

Строгий BSP (Bulk Synchronous Parallel) с Zero-Copy передачей сырой памяти и маппингом ID на стороне отправителя.

### 2.1. Модель: Strict BSP [MVP]

- Шард вычисляет `sync_batch_ticks` (например, 100 тиков = 10 мс) **абсолютно автономно**.
- В конце батча — жёсткий сетевой **барьер**:
  1. Flush исходящих буферов соседям.
  2. Ждём пакеты от всех `neighbors`.
  3. Нельзя начать батч N+1, пока не закрыт батч N.

### 2.2. Latency Hiding (Компенсация Задержек) [MVP]

Сетевой лаг **легализуется** как физическая задержка сигнала (аксона).

- Полученные спайки из батча N не применяются мгновенно.
- Они укладываются в **кольцевой буфер** (`Schedule`) получателя и срабатывают в батче N+1 на тике, равном их `tick_offset`.
- **Результат:** Задержка в 1 батч (10 мс) биологически естественна — сигнал от одной зоны коры до другой идёт 5–20 мс.

### 2.3. Fast Path: Zero-Copy Память [MVP]

Никакого кастомного бинарного протокола с десериализацией. Формат пакета — **плоский дамп памяти GPU**.

**Пайплайн:** `cudaMemcpy (VRAM → RAM)` → TCP/UDP сокет → RAM соседа → аппаратный каст сырых байтов обратно в `&[SpikeEvent]`.

```rust
// Заголовок (8 байт, выровнен)
#[repr(C)]
struct SpikeBatchHeader {
    batch_id: u32,      // Номер батча (защита от рассинхрона)
    spikes_count: u32,  // Количество событий
}

// Тело пакета (ровно 8 байт на спайк — Coalesced Access)
#[repr(C)]
struct SpikeEvent {
    // ПРЯМОЙ индекс в памяти принимающего шарда. Никаких поисков.
    receiver_ghost_id: u32,
    // Смещение внутри батча (0..sync_batch_ticks)
    tick_offset: u8,
    // Паддинг для выравнивания до 64 бит
    _pad: [u8; 3],
}
```

### 2.4. Sender-Side Mapping [MVP]

Принимающий шард **не занимается маршрутизацией** внутри горячего цикла (Hot Loop).

- **Отправитель** при генерации спайка сам переводит свой локальный ID в `receiver_ghost_id` (ID Ghost Axon на целевом сервере). Таблица маппинга формируется при Handshake (создание Ghost Axon).
- **Получатель** берёт `receiver_ghost_id` из пакета и использует как **прямой `u32` индекс** для инъекции в массив. Ноль ветвлений, O(1).

### 2.5. Жизненный Цикл Ghost Axon [MVP: Рождение + Активность, PLANNED: Смерть]

4 фазы: Рождение → Активность → Пропагация → Смерть.

**Шаг 1: Рождение (Slow Path / Night Phase)**

1. Аксон шарда A достигает границы X+. CPU шарда A отправляет `NewAxon { entry_point, vector, type_mask }` в шард B.
2. CPU шарда B в фазу «Ночь» аллоцирует слот в зоне Ghost: `ghost_id = num_local + offset`.
3. Шард B отвечает `Handshake_Ack(local_axon_id_A, ghost_id)`.
4. CPU шарда A записывает маппинг: «Мой аксон X → Шард B, индекс `ghost_id`».

**Шаг 2: Активность (Fast Path / Day Phase)**

1. Аксон шарда A стреляет → CPU формирует `SpikeBatch` с готовым `ghost_id` (Sender-Side Mapping).
2. Шард B получает массив `ghost_indices[]` → ядро `ApplySpikeBatch`: `axon_heads[ghost_id] = 0` за O(1).

**Шаг 3: Пропагация**

`PropagateAxons` безусловно сдвигает Ghost аксоны (`+= v_seg`). `UpdateNeurons` в шарде B проверяет Active Tail через `dist = head - seg_idx` — математика идентична локальным аксонам.

**Шаг 4: Смерть (Night Phase)**

Шард A решает убить аксон (Pruning) → отправляет `PruneAxon(ghost_id)` в шард B. В следующую Ночь шард B освобождает слот и восстанавливает `axon_heads[ghost_id] = AXON_SENTINEL` (0x80000000).

### 2.6. Slow Path: Геометрия [PLANNED]

Передача `AxonHandover` (§1.3) происходит **раз в K батчей** (например, раз в 10–50 мс). Рост аксонов — процесс медленный. Обновлять геометрию каждые 100 мкс не нужно. Это существенно разгружает канал.

### 2.7. Main Loop [MVP: Compute + Network, PLANNED: Geometry]

```python
while simulation_running:
    # 1. Compute Phase (GPU)
    for t in range(sync_batch_ticks):
        current_tick += 1
        apply_remote_spikes(current_tick)   # Из Schedule (Ring Buffer)
        physics_step()
        collect_outgoing_spikes()

    # 2. Network Phase (CPU/IO) — Барьер
    send_buffers_to_neighbors()             # Zero-Copy Flush
    incoming_data = wait_for_neighbors()    # Блокировка

    # 3. Map Phase
    schedule_spikes(incoming_data)           # → Ring Buffer
```

### 2.8. Ring Buffer Schedule [MVP]

Плоский 2D массив — никаких Priority Queue, деревьев или сортировки:

```rust
// Выделяется при старте (на CPU, затем → VRAM)
struct SpikeSchedule {
    // schedule[tick_offset][slot] = ghost_id
    buffer: Vec<Vec<u32>>,      // [sync_batch_ticks][MAX_SPIKES_PER_TICK]
    counts: Vec<u32>,           // Количество спайков в каждом тике
}
```

**Запись (Map Phase, CPU):** Спайки из `SpikeBatch` раскладываются по `tick_offset` из `SpikeEvent`:

```rust
for event in incoming_batch {
    let ghost_id = event.receiver_ghost_id as usize;
    let tick = event.tick_offset as usize;
    
    // Защита от мусорных ID (Инъекция/Баги сети/Десинхронизация)
    // Валидация происходит 1 раз на CPU, разгружая Hot Loop на GPU
    if ghost_id >= local_axons && ghost_id < (local_axons + ghost_axons) && tick < sync_batch_ticks {
        schedule.buffer[tick][schedule.counts[tick]] = ghost_id as u32;
        schedule.counts[tick] += 1;
    } else {
        // Log telemetry: dropped invalid spike (security/corruption warning)
    }
}
```

**Чтение (Day Phase, GPU):** На каждом тике `ApplySpikeBatch` читает **один слот** `schedule[current_tick_in_batch]`, запускается только для `counts[tick]` спайков.

### 2.9. Ping-Pong Double Buffering [MVP]

Чтобы GPU не простаивала на сетевом барьере:

```
Buffer_A ←──── GPU считает Batch N (читает schedule из A)
Buffer_B ←──── Сеть заполняет Batch N+1 (спайки от соседей пишутся в B)

[BARRIER] ───→ swap(A, B) ───→ GPU считает Batch N+1 из B
```

- GPU и NIC работают **параллельно** — GPU не ждёт сеть, пока считает текущий батч.
- На барьере ожидаем только «хвосты» (последние пакеты) и меняем указатели.

### 2.10. Почему Это Сработает [MVP]

| Принцип | Эффект |
|---|---|
| **Масштабируемость** | Платим за латентность сети один раз за 100 тиков, не 100 раз |
| **Детерминизм** | При тех же сидах и `sync_batch_ticks` — бит-в-бит одинаковый результат, независимо от скорости сети |
| **Безопасность** | Если сеть лагает — система притормаживает (Wall Clock Speed падает), но физика (GSOP, спайки) остаётся математически точной |
| **Zero-Copy** | Hot Loop получателя: O(1) вставка по готовому индексу, без десериализации |

---

## 3. Интеграция Атласа (White Matter Routing) [PLANNED]

Атлас (например, на базе FLNe-матриц мармозетки) — **статичная таблица маршрутизации**. Используется только на CPU во время фазы Baking. В горячем цикле (Hot Loop) на GPU Атласа не существует. Пространство между зонами **не симулируется**.

### 3.1. Квотирование (Hard Quotas)

Отказываемся от генерации связей «по вероятности». Используем **жёсткие квоты**.

- Если атлас говорит, что доля V1 во входящих связях V2 составляет 17%, и слою нужно 100 000 входов — Compiler Tool обязан выбрать **ровно 17 000** сом-отправителей из V1.
- Выбор отправителей производится **детерминированным шаффлом** на базе `master_seed` (см. [02_configuration.md §5.3](./02_configuration.md)).

### 3.2. Топографический Маппинг (UV-Projection)

Зоны имеют разные физические размеры. Прямой перенос координат невозможен.

- **UV-нормализация:** Координаты сомы-отправителя переводятся в диапазон `0.0..1.0` (`U`, `V`).
- **Проекция:** Целевая координата = `U × Target_Width`, `V × Target_Height`.
- **Результат:** Пространственная топология проекции сохраняется — то, что рядом в V1, рядом и в V2.

### 3.3. Детерминированное Рассеивание (Jitter)

Никаких проверок коллизий для Ghost Axons на целевой стороне.

- Чтобы аксоны не попадали в один математический пиксель идеальной сетки, к целевой координате добавляется **детерминированный шум:**

```
Target_X += Hash(master_seed + soma_id) % jitter_radius
Target_Y += Hash(master_seed + soma_id + 1) % jitter_radius
```

- Шум воспроизводим при том же `master_seed` — детерминизм сохранён.

### 3.4. Результат: Zero-Cost Routing

1. **Передающий шард (V1):** Создаётся выходной порт (`Port Out`).
2. **Принимающий шард (V2):** Создаётся массив Ghost Axons в рассчитанных координатах (`Target_X`, `Target_Y`). Они начинают **локально прорастать** вглубь целевого слоя, подчиняясь обычной физике Cone Tracing (§4 из [04_connectivity.md](./04_connectivity.md)).
3. **Задержка:** `Delay_Ticks` рассчитывается жёстко по физическому расстоянию между зонами в Атласе. Реальный сетевой лаг (пинг) **прячется** внутри этой математической задержки — сеть перестаёт быть проблемой.

---

## 4. Текущая Реализация в V1 (MVP)

### 4.1. Что Работает Сейчас [MVP]

**Single-Node (Одна GPU):**
- Columnar Memory Layout сохраняется.
- CUDA kernels вычисляют весь день без синхронизации с сетью.
- Шардирование по X/Y отключено — весь мозг на одной GPU VRAM.

**BSP Barrier (Mock):**
- `BspBarrier::new()` создаёт ping-pong буферы (`schedule_a`, `schedule_b`).
- `sync_and_swap()` выполняет жёсткую синхронизацию (swap буферов).
- Сокет **отключен** (`socket: None`) — для offline тестов.

**SpikeSchedule:**
```rust
pub struct SpikeSchedule {
    pub sync_batch_ticks: usize,
    pub buffer: Vec<u32>,      // [sync_batch_ticks × MAX_SPIKES_PER_TICK]
    pub counts: Vec<u32>,      // [sync_batch_ticks]
}
```
- Flat 1D array для Single DMA to VRAM.
- Каждый тик: GPU берёт `buffer[tick * MAX_SPIKES_PER_TICK .. +counts[tick]]`.
- Map Phase (CPU) раскладывает спайки по `tick_offset` из `SpikeEvent`.

**GhostConnection (Metadata):**
```rust
pub struct GhostConnection {
    pub local_axon_id: u32,     // Индекс в axon_heads[] на целевом шарде
    pub paired_src_soma: u32,   // Soma ID на источнике (для трассировки)
}
```
- Загружается из файла `.ghosts` (Baker output).
- Используется только для трансляции soma_id → ghost_id при спайке.

### 4.2. Что НЕ Работает Пока [PLANNED]

| Функция | Статус | Причина |
|---|---|---|
| **Multi-Node (Network)** | PLANNED | Сокет не подключен, одна GPU |
| **Dynamic AxonHandover** | PLANNED | Аксоны не растут через границы; они бакутся заранее |
| **Periodic Boundaries (Toroidal)** | PLANNED | Замыкание Z и кольцо X/Y требуют изменения конфига |
| **Slow Path Geometry Sync** | PLANNED | Геометрия вычисляется только раз при Baking |
| **Pruning & Night Phase Networking** | PLANNED | Удаление Ghost Axons на целевом шарде не реализовано |
| **Latency Hiding Variance** | PLANNED | Фиксированная задержка 1 батч; адаптивная задержка не реализована |
| **Atlas-Based Routing** | PLANNED | Маршрутизация между зонами фиксирована в `.ghosts` файлах |

---

## 5. Структуры Данных (Реальные Сигнатуры)

### 5.1. SpikeBatchHeader & SpikeEvent

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeBatchHeader {
    pub batch_id: u32,      // Номер батча (защита от рассинхрона)
    pub spikes_count: u32,  // Количество событий в батче
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeEvent {
    pub receiver_ghost_id: u32,  // Ghost Axon ID на целевом шарде (прямой индекс)
    pub tick_offset: u8,         // Тик внутри батча (0..sync_batch_ticks)
    pub _pad: [u8; 3],           // Выравнивание до 8 байт
}
```

**Размер:** 8 байт на спайк (Coalesced Access на GPU).

**Передача:** Заголовок (8 байт) + массив SpikeEvent (8 байт × count).

### 5.2. BspBarrier Internals

```rust
pub struct BspBarrier {
    pub schedule_a: SpikeSchedule,              // Буфер 1
    pub schedule_b: SpikeSchedule,              // Буфер 2
    pub writing_to_b: bool,                     // Куда пишет сеть
    pub outgoing_batches: HashMap<u32, Vec<SpikeEvent>>,  // Исходящие спайки по шардам
    pub socket: Option<NodeSocket>,             // Сокет (None в MVP)
    pub peer_addresses: HashMap<u32, SocketAddr>,  // Маршрут: shard_id → IP:port
}
```

**Жизненный цикл sync_and_swap():**
1. Flip `writing_to_b` (GPU читает из одного, сеть пишет в другой).
2. Если `socket.is_some()`: отправить `outgoing_batches` всем соседям (async).
3. Ждём прихода батчей от соседей (await).
4. `ingest_spike_batch()` раскладывает входящие спайки в текущий Schedule.
5. Сброс Schedule на следующий батч.

### 5.3. GhostConnection File Format

Файл `{shard_name}.ghosts` (бинарный, Little-Endian):

```
[0..3]   magic: u32 = 0x47485354 ("GHST")
[4]      version: u8 = 1
[5]      padding: u8
[6..7]   width: u16 (измерение X/Y стыка)
[8..9]   height: u16 (измерение Z или второе строящееся направление)
[10..11] padding: u16
[12..15] src_zone_hash: u32 (идентификатор Zona-источника)
[16..19] dst_zone_hash: u32 (идентификатор Zona-приёма)
[20..]   connections: [GhostConnection; width × height]
         Каждый: local_axon_id (u32) + paired_src_soma (u32) = 8 байт
```

**Примечание:** Массив Connections отсортирован по (Y, Z) если X-граница или (X, Z) если Y-граница. Это позволяет GPU применять `ApplySpikeBatch` с хорошей локальностью доступа.

---

## Связанные документы

| Документ | Что связывается |
|---|---|
| [04_connectivity.md](./04_connectivity.md) | §1.7: Ghost Axons, Cone Tracing, Axon Growth |
| [05_signal_physics.md](./05_signal_physics.md) | §1.2.1: ApplySpikeBatch kernel, Ghost indices |
| [07_gpu_runtime.md](./07_gpu_runtime.md) | §2: Batch Orchestration, sync_and_swap timing, Stream management |
| [09_baking_pipeline.md](./09_baking_pipeline.md) | §3: Ghost Connection generation, Z-sort for boundaries |
| [02_configuration.md](./02_configuration.md) | §5.3: master_seed, deterministic topology |
| [project_structure.md](../project_structure.md) | Distributed arch role in overall Genesis design |

## 6. [TODO] Autonomous Node Recovery (Fault Tolerance)

**Goal**: Ensure high availability and fault tolerance for the distributed simulation network.

### Proposed Architecture
- **State Replication**: Each node will maintain redundant backups of its VRAM state / graph topology checkpoints in multiple distinct storage locations.
- **Heartbeat & Consensus**: Implement a network-wide gossip or heartbeat protocol so that all nodes constantly monitor the health and responsiveness of their peers.
- **Self-Healing (Autonomous Orchestration)**: If a node failure is detected, the network will automatically identify available free hardware, deploy a new genesis-runtime instance, restore from the latest distributed backup, and dynamically re-route intra-gpu/network channels to seamlessly patch the gap without human intervention.

---

## Changelog

| Дата | Версия | Описание изменений |
|---|---|---|
| 2026-03-02 | 1.2 | Добавлено требование по Autonomous Node Recovery (Fault Tolerance) |
| 2026-02-28 | 1.1 | Разделение на [MVP] vs [PLANNED] маркеры. Уточнение GhostConnection (V1 реальность). Обновлены SpikeEvent, BspBarrier, GhostConnection структуры согласно реальному коду. Добавлен раздел "Текущая реализация в V1" и таблица функций. |
| TBD | 1.0 | Первая версия спеки |


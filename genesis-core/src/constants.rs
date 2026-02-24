/// Sentinel value for inactive axons.
/// dist = AXON_SENTINEL - seg_idx = huge → is_active = false (u32 underflow legalized).
pub const AXON_SENTINEL: u32 = 0x80000000;

/// Maximum dendrite slots per neuron (LTM: 0..79, WM: 80..127).
pub const MAX_DENDRITE_SLOTS: usize = 128;

/// LTM / WM boundary slot index.
pub const WM_SLOT_START: usize = 80;

/// target_packed bit layout: [31..10] Axon_ID (22 bits) | [9..0] Segment_Index (10 bits)
pub const TARGET_AXON_SHIFT: u32 = 10;
pub const TARGET_SEG_MASK: u32 = 0x3FF;

/// Warp size for GPU alignment (padded_n must be multiple of this).
pub const GPU_WARP_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Физические константы (Spec 01 §1.6) — Фиксированная конфигурация
// Изменение любого из них требует пересчёта V_SEG и проверки компилятора.
// ---------------------------------------------------------------------------

/// Шаг времени: 100 мкс = 0.1 мс.
pub const TICK_DURATION_US: u32 = 100;

/// Размер вокселя в мкм.
pub const VOXEL_SIZE_UM: u32 = 25;

/// Длина одного сегмента аксона в вокселях.
pub const SEGMENT_LENGTH_VOXELS: u32 = 2;

/// Длина сегмента в мкм (= VOXEL_SIZE_UM × SEGMENT_LENGTH_VOXELS).
pub const SEGMENT_LENGTH_UM: u32 = VOXEL_SIZE_UM * SEGMENT_LENGTH_VOXELS; // 50

/// Скорость сигнала в мкм/тик (0.5 м/с = 50 мкм/тик).
pub const SIGNAL_SPEED_UM_TICK: u32 = 50;

/// Дискретная скорость: сегментов за тик. Обязана быть целым числом.
pub const V_SEG: u32 = SIGNAL_SPEED_UM_TICK / SEGMENT_LENGTH_UM; // 1

/// Инвариант §1.6: signal_speed_um_tick ОБЯЗАНА делиться на segment_length_um без остатка.
/// Если v_seg дробное — GPU не может работать без флоатов — нарушение Integer Physics.
#[allow(clippy::eq_op)]
const _: () = assert!(
    SIGNAL_SPEED_UM_TICK % SEGMENT_LENGTH_UM == 0,
    "Spec 01 §1.6 violation: signal_speed_um_tick must be divisible by segment_length_um (v_seg must be integer)"
);

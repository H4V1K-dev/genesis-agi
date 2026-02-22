use genesis_core::{
    constants::{AXON_SENTINEL, MAX_DENDRITE_SLOTS},
    layout::padded_n,
};

/// Бинарное представление SoA-состояния шарда.
/// Байт-в-байт совпадает с VRAM layout для `cudaMemcpy`.
/// (02_configuration.md §2.4 ShardStateSoA)
pub struct ShardStateSoA {
    /// Количество нейронов (выровнено до кратного 32 — padded_n).
    pub padded_n: usize,
    /// Количество аксонов (local + ghost + virtual, выровнено до 32).
    pub total_axons: usize,

    // --- Soma arrays [padded_n] ---
    /// Мембранный потенциал (i32, microVolts). Начальное = rest_potential.
    pub voltage: Vec<i32>,
    /// Флаги нейрона: [7:6]=variant_id, [5]=is_spiking, [3:0]=type_mask.
    pub flags: Vec<u8>,
    /// Адаптивный порог = base_threshold + threshold_offset (i32).
    pub threshold_offset: Vec<i32>,
    /// Счётчик рефрактерности (u8, тики).
    pub refractory_counter: Vec<u8>,
    /// Маппинг soma_id → local axon_id.
    pub soma_to_axon: Vec<u32>,

    // --- Dendrite arrays — Columnar Layout [MAX_SLOTS × padded_n] ---
    // Обращение: data[slot * padded_n + neuron_id]
    /// Packed target: upper 24b = axon_id, lower 8b = segment offset. 0 = empty.
    pub dendrite_targets: Vec<u32>,
    /// Synaptic weights i16. Sign = excitatory(+) / inhibitory(-).
    pub dendrite_weights: Vec<i16>,
    /// Synapse refractory counters (u8, тики).
    pub dendrite_timers: Vec<u8>,

    // --- Axon arrays [total_axons] ---
    /// Текущая позиция головы аксона. AXON_SENTINEL = неактивен.
    pub axon_heads: Vec<u32>,
}

impl ShardStateSoA {
    /// Создаёт пустой (инициализированный в покой) шард.
    ///
    /// - `neuron_count` — реальное (не выровненное) число нейронов
    /// - `axon_count`   — реальное число аксонов
    /// - `rest_potential` — начальный voltage для всех нейронов
    pub fn new_blank(neuron_count: usize, axon_count: usize, rest_potential: i32) -> Self {
        let pn = padded_n(neuron_count);
        let pa = padded_n(axon_count);
        let dendrite_cells = MAX_DENDRITE_SLOTS * pn;

        Self {
            padded_n: pn,
            total_axons: pa,

            voltage: vec![rest_potential; pn],
            flags: vec![0u8; pn],
            threshold_offset: vec![0i32; pn],
            refractory_counter: vec![0u8; pn],
            soma_to_axon: vec![u32::MAX; pn], // u32::MAX = нет аксона

            // Dendrite columnar — всё пусто (target=0, weight=0, timer=0)
            dendrite_targets: vec![0u32; dendrite_cells],
            dendrite_weights: vec![0i16; dendrite_cells],
            dendrite_timers: vec![0u8; dendrite_cells],

            // Все аксоны — SENTINEL (неактивны, сеть не выстрелит в тик 0)
            axon_heads: vec![AXON_SENTINEL; pa],
        }
    }

    /// Общий размер данных в байтах (для проверки перед cudaMemcpy).
    pub fn byte_size(&self) -> usize {
        let pn = self.padded_n;
        let pa = self.total_axons;
        let dc = MAX_DENDRITE_SLOTS * pn;

        pn * 4  // voltage i32
        + pn    // flags u8
        + pn * 4  // threshold_offset i32
        + pn    // refractory_counter u8
        + pn * 4  // soma_to_axon u32
        + dc * 4  // dendrite_targets u32
        + dc * 2  // dendrite_weights i16
        + dc    // dendrite_timers u8
        + pa * 4 // axon_heads u32
    }

    /// Сериализует SoA в плоский байтовый вектор — готов к записи в `.state`.
    /// Порядок массивов: voltage → flags → threshold_offset → refractory_counter
    ///                   → soma_to_axon → dendrite_targets → dendrite_weights
    ///                   → dendrite_timers → axon_heads
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_size());

        for &v in &self.voltage {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.flags);
        for &v in &self.threshold_offset {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.refractory_counter);
        for &v in &self.soma_to_axon {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.dendrite_targets {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.dendrite_weights {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.dendrite_timers);
        for &v in &self.axon_heads {
            out.extend_from_slice(&v.to_le_bytes());
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::constants::AXON_SENTINEL;

    #[test]
    fn blank_shard_axons_are_sentinel() {
        let shard = ShardStateSoA::new_blank(100, 100, 10_000);
        assert!(
            shard.axon_heads.iter().all(|&h| h == AXON_SENTINEL),
            "All axon_heads must be AXON_SENTINEL at init"
        );
    }

    #[test]
    fn blank_shard_dendrites_are_empty() {
        let shard = ShardStateSoA::new_blank(64, 64, 10_000);
        assert!(
            shard.dendrite_targets.iter().all(|&t| t == 0),
            "All dendrite_targets must be 0 (empty) at init"
        );
    }

    #[test]
    fn padded_n_alignment() {
        // 100 neurons → padded to 128
        let shard = ShardStateSoA::new_blank(100, 100, 10_000);
        assert_eq!(shard.padded_n, 128);
        assert_eq!(shard.voltage.len(), 128);
        assert_eq!(shard.dendrite_targets.len(), 128 * MAX_DENDRITE_SLOTS);
    }

    #[test]
    fn byte_size_matches_to_bytes() {
        let shard = ShardStateSoA::new_blank(32, 32, 10_000);
        let bytes = shard.to_bytes();
        assert_eq!(bytes.len(), shard.byte_size());
    }

    #[test]
    fn voltage_encoded_correctly() {
        let rest = 10_000i32;
        let shard = ShardStateSoA::new_blank(32, 32, rest);
        let bytes = shard.to_bytes();
        // Первые 4 байта = voltage[0] = rest_potential в LE
        let v = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(v, rest);
    }
}

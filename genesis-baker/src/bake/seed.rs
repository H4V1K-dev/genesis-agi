/// Детерминированное хэширование entity по `master_seed`.
/// Используется везде где нужен воспроизводимый "случайный" результат для конкретного нейрона.
/// Алгоритм: wyhash (02_configuration.md §2.4)
///
/// Дефолтный master_seed: "GENESIS"
#[allow(dead_code)]
pub const DEFAULT_MASTER_SEED: &str = "GENESIS";

/// Хэшируем строку-сид в u64.
pub fn seed_from_str(s: &str) -> u64 {
    wyhash::wyhash(s.as_bytes(), 0)
}

/// Сид для конкретного entity (нейрона/аксона) по его packed_position.
pub fn entity_seed(master_seed: u64, packed_pos: u32) -> u64 {
    wyhash::wyhash(&packed_pos.to_le_bytes(), master_seed)
}

/// Быстрый псевдослучайный float [0.0, 1.0) из seed.
pub fn random_f32(seed: u64) -> f32 {
    // Берём старшие 23 бита и строим float в [1.0, 2.0), затем смещаем в [0.0, 1.0)
    let bits = (seed >> 41) as u32 | 0x3F800000;
    f32::from_bits(bits) - 1.0
}

/// Shuffle индексов [0..len) детерминированно через Fisher-Yates с wyhash.
pub fn shuffle_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut s = seed;
    for i in (1..len).rev() {
        s = wyhash::wyhash(&s.to_le_bytes(), s);
        let j = (s as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_seed_stable() {
        // Один и тот же вход → один и тот же хэш всегда
        let s1 = seed_from_str(DEFAULT_MASTER_SEED);
        let s2 = seed_from_str(DEFAULT_MASTER_SEED);
        assert_eq!(s1, s2);
        assert_ne!(s1, 0);
    }

    #[test]
    fn entity_seed_unique_per_position() {
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let s0 = entity_seed(master, 0);
        let s1 = entity_seed(master, 1);
        assert_ne!(s0, s1);
    }

    #[test]
    fn random_f32_in_range() {
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        for i in 0..100u32 {
            let f = random_f32(entity_seed(master, i));
            assert!(f >= 0.0 && f < 1.0, "f={} out of [0,1)", f);
        }
    }

    #[test]
    fn shuffle_is_deterministic() {
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let a = shuffle_indices(100, master);
        let b = shuffle_indices(100, master);
        assert_eq!(a, b);
        // И не тривиальный (0,1,2,...) порядок
        assert_ne!(a[0], 0, "shuffle should not be identity");
    }
}

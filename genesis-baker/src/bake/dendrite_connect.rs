use crate::bake::layout::ShardSoA;
use crate::bake::axon_growth::GrownAxon;
use genesis_core::types::PackedPosition;
use genesis_core::config::blueprints::NeuronType;
use crate::bake::spatial_grid::SpatialGrid;

pub fn connect_dendrites(
    shard: &mut ShardSoA,
    positions: &[PackedPosition],
    axons: &[GrownAxon],
    types: &[NeuronType],
    _master_seed: u64,
    cell_size: u32,
) -> usize {
    // 1. Строим пространственный хэш по сомам для быстрого поиска
    let grid = SpatialGrid::new(positions.to_vec(), cell_size);
    let mut soma_slot_counts = vec![0usize; shard.padded_n];
    let mut total_synapses = 0;

    // 2. Аксон вещает (Publisher), дендриты подписываются (Subscribers)
    for (axon_id, axon) in axons.iter().enumerate() {
        let owner_type = &types[axon.type_idx];
        let owner_name = &owner_type.name;

        for (seg_idx, &packed_seg) in axon.segments.iter().enumerate() {
            let seg_pos = PackedPosition(packed_seg);

            // Zero-Allocation сканирование соседей в радиусе 1 чанка
            grid.for_each_in_radius(&seg_pos, 1, |soma_dense_id| {
                let soma_id = soma_dense_id as usize;
                let current_slots = soma_slot_counts[soma_id];

                // Hard Limit: 128 дендритов
                if current_slots >= genesis_core::constants::MAX_DENDRITE_SLOTS { return; }
                
                // Самоисключение: не коннектимся к собственному аксону
                if axon.soma_idx == soma_id { return; } 

                let target_type = &types[positions[soma_id].type_id() as usize];

                // Фильтр совместимости (Dendrite Whitelist)
                if !target_type.dendrite_whitelist.is_empty() && !target_type.dendrite_whitelist.contains(owner_name) {
                    return; 
                }

                // Rule of Uniqueness: O(K) линейный поиск (K <= 128, всё лежит в L1)
                let mut is_duplicate = false;
                for s in 0..current_slots {
                    let col_idx = s * shard.padded_n + soma_id;
                    let target = shard.dendrite_targets[col_idx];
                    if genesis_core::layout::unpack_axon_id(target) == axon_id as u32 {
                        is_duplicate = true;
                        break;
                    }
                }
                if is_duplicate { return; }

                // Запись синапса (Columnar SoA Layout)
                let col_idx = current_slots * shard.padded_n + soma_id;
                shard.dendrite_targets[col_idx] = genesis_core::layout::pack_dendrite_target(axon_id as u32, seg_idx as u32);

                // Dale's Law: Знак берём из принимающей стороны (по спецификации V1)
                let weight = target_type.initial_synapse_weight as i16;
                shard.dendrite_weights[col_idx] = if target_type.is_inhibitory { -weight } else { weight };

                soma_slot_counts[soma_id] += 1;
                total_synapses += 1;
            });
        }
    }
    total_synapses
}

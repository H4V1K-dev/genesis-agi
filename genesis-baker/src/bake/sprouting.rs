use genesis_core::config::blueprints::NeuronType;



/// Вычисляет привлекательность сомы-кандидата для растущего аксона.
/// Вся математика здесь легально использует f32, так как это Night Phase.
#[inline]
pub fn compute_sprouting_score(
    target_type: &NeuronType,
    distance: f32,
    power_index: f32,
    noise: f32,
    owner_type_id: u8,
    target_type_id: u8,
) -> f32 {
    let dist_score = 1.0 / (distance + 1.0);
    
    // [DOD] Branchless Affinity
    let is_same = (owner_type_id == target_type_id) as i32 as f32;
    let affinity_mod = (is_same * target_type.type_affinity 
                      + (1.0 - is_same) * (1.0 - target_type.type_affinity)) * 2.0;

    let mut score = dist_score * target_type.sprouting_weight_distance 
        + power_index * target_type.sprouting_weight_power 
        + noise * target_type.sprouting_weight_explore;
        
    score *= affinity_mod;
    score
}

/// Евклидово расстояние в вокселях между двумя точками.
pub fn voxel_dist(ax: u32, ay: u32, az: u32, bx: u32, by: u32, bz: u32) -> f32 {
    let dx = ax as f32 - bx as f32;
    let dy = ay as f32 - by as f32;
    let dz = az as f32 - bz as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

use genesis_core::layout::pack_dendrite_target;
use genesis_core::constants::MAX_DENDRITE_SLOTS;

/// CPU Sprouting Pass — заполняет пустые дендритные слоты.
/// Zero-copy: работает напрямую со слайсами из SHM.
pub fn run_sprouting_pass(
    targets: &mut [u32],
    weights: &mut [i16],
    padded_n: usize,
    types: &[NeuronType],
    grid: &crate::bake::spatial_grid::AxonSegmentGrid,
    positions: &[genesis_core::types::PackedPosition],
    epoch: u64,
    _axon_types: &[u8],
    whitelist_masks: &[u16; 16],
    voxel_size_um: f32,
) -> usize {
    let mut new_synapses = 0;



    for i in 0..padded_n {
        let my_pos = positions[i];
        if my_pos.0 == 0 { continue; }
        
        let my_type_id = my_pos.type_id() as usize;
        let my_type = &types[my_type_id];
        let radius_vox = (my_type.dendrite_radius_um / voxel_size_um).ceil() as u32; 

        for slot in (0..MAX_DENDRITE_SLOTS).rev() {
            let col_idx = slot * padded_n + i;
            if targets[col_idx] != 0 {
                break;
            }

            let salt = (i as u32).wrapping_add(slot as u32).wrapping_add(1);
            let hash = fnv1a(epoch, salt);
            
            if let Some(seg_ref) = grid.get_random_candidate(my_pos.x() as u32, my_pos.y() as u32, my_pos.z() as u32, radius_vox, hash) {
                let candidate_axon_id = seg_ref.axon_id;
                
                // Self-exclusion
                // We'd need soma_to_axon or similar here, but for now we skip self-check in run_sprouting_pass 
                // as it's a "quick" pass. Refinement:
                // if candidate_axon_id == i as u32 { continue; }

                let target_type_id = seg_ref.type_idx as usize;
                
                // Whitelist Filter
                let mask = whitelist_masks[(my_type_id % 16) as usize];
                if mask != 0xFFFF && (mask & (1 << (target_type_id % 16))) == 0 {
                    continue;
                }

                let new_target = pack_dendrite_target(candidate_axon_id, seg_ref.seg_idx as u32);
                
                // [DOD Stage 46.3] Dale's Law from AXON OWNER
                let owner_type = &types[target_type_id % types.len()];
                let abs_w = owner_type.initial_synapse_weight as i16;
                let final_weight = if owner_type.is_inhibitory { -abs_w } else { abs_w };

                targets[col_idx] = new_target;
                weights[col_idx] = final_weight;
                new_synapses += 1;
                break;
            }
        }
    }

    new_synapses
}

/// FNV-1a Stateless Hash (Инвариант #7 — детерминизм через seed+id)
fn fnv1a(seed: u64, salt: u32) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &b in &seed.to_le_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for &b in &salt.to_le_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}


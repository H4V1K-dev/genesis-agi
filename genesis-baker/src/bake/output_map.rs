// genesis-baker/src/bake/output_map.rs
use genesis_core::constants::GXO_MAGIC;
use genesis_core::config::IoConfig;
use genesis_core::hash::fnv1a_32;
use std::collections::HashMap;
use std::path::Path;

pub struct BakedOutputs {
    pub gxo_binary: Vec<u8>,
    pub num_mapped_somas: u32,
}

/// Запекает .gxo файл используя Z-Sort алгоритм для выбора сом-кандидатов.
pub fn bake_outputs(
    out_dir: &Path,
    io_config: &IoConfig,
    zone_width_um: f32,
    zone_depth_um: f32,
    neurons_packed_pos: &[u32],
    name_map: &HashMap<String, u8>,
) {
    let result = bake_outputs_to_memory(io_config, zone_width_um, zone_depth_um, neurons_packed_pos, name_map);
    if result.gxo_binary.is_empty() {
        return;
    }

    let path = out_dir.join("shard.gxo");
    std::fs::write(path, result.gxo_binary).expect("Failed to write .gxo file");
}

pub fn bake_outputs_to_memory(
    io_config: &IoConfig,
    zone_width_um: f32,
    zone_depth_um: f32,
    neurons_packed_pos: &[u32],
    name_map: &HashMap<String, u8>,
) -> BakedOutputs {
    if io_config.outputs.is_empty() {
        return BakedOutputs { gxo_binary: vec![], num_mapped_somas: 0 };
    }

    let mut total_pixels = 0;
    for m in &io_config.outputs {
        total_pixels += m.width * m.height;
    }

    let mut payload_soma_ids = vec![0u32; total_pixels as usize];
    let mut current_offset = 0;
    let mut num_mapped_somas = 0;

    for matrix in &io_config.outputs {
        let pixels = matrix.width * matrix.height;
        
        for py in 0..matrix.height {
            for px in 0..matrix.width {
                let x_min = (px as f32 / matrix.width as f32) * zone_width_um;
                let x_max = ((px + 1) as f32 / matrix.width as f32) * zone_width_um;
                let y_min = (py as f32 / matrix.height as f32) * zone_depth_um;
                let y_max = ((py + 1) as f32 / matrix.height as f32) * zone_depth_um;

                let mut best_soma_id = u32::MAX;
                let mut min_z = u32::MAX;

                let target_type_idx = if !matrix.target_type.is_empty() {
                    name_map.get(&matrix.target_type).cloned()
                } else {
                    None
                };

                for (dense_id, &packed) in neurons_packed_pos.iter().enumerate() {
                    // Type filter
                    if let Some(needed_idx) = target_type_idx {
                        let type_idx = (packed >> 28) as u8;
                        if type_idx != needed_idx {
                            continue;
                        }
                    }

                    let vx = (packed & 0x3FF) as f32; 
                    let vy = ((packed >> 10) & 0x3FF) as f32; 
                    let vz = (packed >> 20) & 0xFF;

                    if vx >= x_min && vx < x_max && vy >= y_min && vy < y_max {
                        if vz < min_z {
                            min_z = vz;
                            best_soma_id = dense_id as u32;
                        }
                    }
                }

                if best_soma_id != u32::MAX {
                    num_mapped_somas += 1;
                } else {
                    best_soma_id = 0; 
                }

                let pixel_idx = (py * matrix.width) + px;
                payload_soma_ids[(current_offset + pixel_idx) as usize] = best_soma_id;
            }
        }
        current_offset += pixels;
    }

    let mut gxo_binary = Vec::new();
    // Header (12 bytes)
    gxo_binary.extend_from_slice(&GXO_MAGIC.to_le_bytes());
    gxo_binary.extend_from_slice(&[1u8, 0u8]); // Version 1, Padding 1
    gxo_binary.extend_from_slice(&(io_config.outputs.len() as u16).to_le_bytes());
    gxo_binary.extend_from_slice(&total_pixels.to_le_bytes());

    let mut current_offset_px: u32 = 0;
    for m in &io_config.outputs {
        let name_hash = fnv1a_32(m.name.as_bytes());
        gxo_binary.extend_from_slice(&name_hash.to_le_bytes());
        gxo_binary.extend_from_slice(&current_offset_px.to_le_bytes());
        gxo_binary.extend_from_slice(&(m.width as u16).to_le_bytes());
        gxo_binary.extend_from_slice(&(m.height as u16).to_le_bytes());
        gxo_binary.extend_from_slice(&(m.stride as u8).to_le_bytes());
        gxo_binary.extend_from_slice(&[0, 0, 0]); // Padding (3 bytes)
        
        current_offset_px += m.width * m.height;
    }

    let payload_bytes = unsafe {
        std::slice::from_raw_parts(
            payload_soma_ids.as_ptr() as *const u8,
            payload_soma_ids.len() * 4
        )
    };
    gxo_binary.extend_from_slice(payload_bytes);

    BakedOutputs { gxo_binary, num_mapped_somas }
}

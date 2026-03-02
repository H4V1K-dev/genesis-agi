// genesis-baker/src/bake/input_map.rs
use crate::bake::axon_growth::GrownAxon;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use genesis_core::constants::GXI_MAGIC;
use genesis_core::config::IoConfig;
use genesis_core::hash::fnv1a_32;

/// Выращивает виртуальные аксоны для входных данных и генерирует GXI-файл.
pub fn bake_inputs(
    out_dir: &Path,
    io_config: &IoConfig,
    base_axon_id: u32,
) -> Vec<GrownAxon> {
    let mut total_pixels = 0;
    for m in &io_config.inputs {
        total_pixels += m.width * m.height;
    }

    if total_pixels == 0 {
        return vec![];
    }

    let mut payload_axon_ids = vec![0u32; total_pixels as usize];
    let mut virtual_axons = Vec::with_capacity(total_pixels as usize);

    for i in 0..total_pixels {
        payload_axon_ids[i as usize] = base_axon_id + i;
        
        // Push a lobotomized virtual axon. No physical geometry needed.
        virtual_axons.push(GrownAxon {
            soma_idx: usize::MAX, // Mark as external / virtual
            type_idx: 0,          // Default excitatory
            tip_x: 0,
            tip_y: 0,
            tip_z: 0,
            length_segments: 0,
            segments: vec![],
            last_dir: glam::Vec3::ZERO,
        });
    }

    write_gxi_binary(out_dir, io_config, total_pixels, &payload_axon_ids);

    virtual_axons
}

fn write_gxi_binary(
    out_dir: &Path,
    io_config: &IoConfig,
    total_pixels: u32,
    payload: &[u32]
) {
    let path = out_dir.join("shard.gxi");
    let mut file = BufWriter::new(File::create(path).expect("Failed to create .gxi file"));

    // Header (12 bytes)
    file.write_all(&GXI_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&[1u8, 0u8]).unwrap(); // Version 1, Padding 0
    file.write_all(&(io_config.inputs.len() as u16).to_le_bytes()).unwrap();
    file.write_all(&total_pixels.to_le_bytes()).unwrap();

    let mut current_offset: u32 = 0;
    for m in &io_config.inputs {
        let name_hash = fnv1a_32(m.name.as_bytes());
        file.write_all(&name_hash.to_le_bytes()).unwrap();
        file.write_all(&current_offset.to_le_bytes()).unwrap();
        file.write_all(&(m.width as u16).to_le_bytes()).unwrap();
        file.write_all(&(m.height as u16).to_le_bytes()).unwrap();
        file.write_all(&(m.stride as u8).to_le_bytes()).unwrap();
        file.write_all(&[0, 0, 0]).unwrap(); // Padding (3 bytes)
        
        current_offset += m.width * m.height;
    }

    // Payload
    let payload_bytes = unsafe {
        std::slice::from_raw_parts(
            payload.as_ptr() as *const u8,
            payload.len() * 4
        )
    };
    file.write_all(payload_bytes).unwrap();
}

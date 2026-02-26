use genesis_core::config::{SimulationConfig, BlueprintsConfig, IoConfig};
use crate::bake::neuron_placement::PlacedNeuron;
use genesis_core::coords::unpack_position;

const GXO_MAGIC: u32 = 0x47584F30; // "GXO0"
const GXO_VERSION: u16 = 1;

pub struct BakedOutputMap {
    pub gxo_binary: Vec<u8>,
    pub num_mapped_somas: u32,
}

pub fn bake_output_maps(
    io_config: &IoConfig,
    zone_name: &str,
    placed_neurons: &[PlacedNeuron],
    blueprints: &BlueprintsConfig,
    sim: &SimulationConfig,
) -> BakedOutputMap {
    let outputs: Vec<_> = io_config.outputs.iter()
        .filter(|o| o.source_zone == zone_name)
        .collect();

    if outputs.is_empty() {
        return BakedOutputMap {
            gxo_binary: vec![],
            num_mapped_somas: 0,
        };
    }

    let world_w_vox = sim.world.width_um / sim.simulation.voxel_size_um;
    let world_d_vox = sim.world.depth_um / sim.simulation.voxel_size_um;

    // Total tiles to compute proportional height
    let total_area_tiles: u32 = outputs.iter().map(|o| o.width * o.height).sum();
    let mut current_y_vox = 0.0;

    // We will build the GXO format piece by piece
    // Header
    let mut gxo_bytes = Vec::new();
    gxo_bytes.extend_from_slice(&GXO_MAGIC.to_le_bytes());
    gxo_bytes.extend_from_slice(&GXO_VERSION.to_le_bytes());
    gxo_bytes.extend_from_slice(&(outputs.len() as u16).to_le_bytes());
    // Placeholders for total_somas and readout_batch_ticks
    let total_somas_offset = gxo_bytes.len();
    gxo_bytes.extend_from_slice(&0u32.to_le_bytes()); // total_somas
    
    let default_batch_ticks = sim.simulation.sync_batch_ticks as u32;
    let batch_ticks = io_config.readout_batch_ticks.unwrap_or(default_batch_ticks);
    gxo_bytes.extend_from_slice(&batch_ticks.to_le_bytes());

    let mut map_headers = Vec::new();
    let mut pixel_indices = Vec::new(); // (soma_offset, soma_count)
    let mut all_somas = Vec::new();

    for map in &outputs {
        // Strip allocation
        let fraction = (map.width * map.height) as f32 / total_area_tiles as f32;
        let strip_d_vox = world_d_vox as f32 * fraction;
        let strip_y_start = current_y_vox;
        current_y_vox += strip_d_vox;

        let mut name_bytes = [0u8; 32];
        let name_slice = map.name.as_bytes();
        let copy_len = name_slice.len().min(32);
        name_bytes[..copy_len].copy_from_slice(&name_slice[..copy_len]);

        let map_somas_start = all_somas.len() as u32;

        let tile_w_vox = world_w_vox as f32 / map.width as f32;
        let tile_d_vox = strip_d_vox / map.height as f32;

        for row in 0..map.height {
            for col in 0..map.width {
                let rx = (col as f32 * tile_w_vox) as u32;
                let ry = strip_y_start as u32 + (row as f32 * tile_d_vox) as u32;
                let rx_end = ((col + 1) as f32 * tile_w_vox) as u32;
                let ry_end = strip_y_start as u32 + ((row + 1) as f32 * tile_d_vox) as u32;

                let pixel_somas_start = all_somas.len() as u32;
                let mut pixel_somas_count = 0u16;

                // Find all somas in this tile
                for (idx, neuron) in placed_neurons.iter().enumerate() {
                    let (nx, ny, _nz, ntype) = unpack_position(neuron.position);
                    if nx >= rx && nx < rx_end && ny >= ry && ny < ry_end {
                        let is_match = map.target_type == "ALL" || 
                            blueprints.neuron_types.get(ntype as usize)
                            .map_or(false, |nt| nt.name == map.target_type);
                        
                        if is_match {
                            all_somas.push(idx as u32);
                            pixel_somas_count += 1;
                        }
                    }
                }

                pixel_indices.push((pixel_somas_start, pixel_somas_count));
            }
        }

        let map_somas_count = all_somas.len() as u32 - map_somas_start;

        let mut header = Vec::new();
        header.extend_from_slice(&name_bytes);
        header.extend_from_slice(&(map.width as u16).to_le_bytes());
        header.extend_from_slice(&(map.height as u16).to_le_bytes());
        header.extend_from_slice(&map_somas_start.to_le_bytes());
        header.extend_from_slice(&map_somas_count.to_le_bytes());
        
        map_headers.push(header);
    }

    let total_somas = all_somas.len() as u32;

    // Update total_somas in header
    gxo_bytes[total_somas_offset..total_somas_offset+4].copy_from_slice(&total_somas.to_le_bytes());

    // Assemble final GXO buffer
    for hdr in map_headers {
        gxo_bytes.extend(hdr);
    }

    for (offset, count) in pixel_indices {
        gxo_bytes.extend_from_slice(&offset.to_le_bytes());
        gxo_bytes.extend_from_slice(&count.to_le_bytes());
    }

    for soma_id in all_somas {
        gxo_bytes.extend_from_slice(&soma_id.to_le_bytes());
    }

    BakedOutputMap {
        gxo_binary: gxo_bytes,
        num_mapped_somas: total_somas,
    }
}

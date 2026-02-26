use std::fs::File;
use std::io::Read;
use std::path::Path;
use anyhow::{Context, Result};

pub struct GxoMapDescriptor {
    pub name: String,
    pub width: u16,
    pub height: u16,
    pub somas_offset: u32,
    pub somas_count: u32,
}

pub struct GxoPixelIndex {
    pub offset: u32,
    pub count: u16,
}

pub struct GxoFile {
    pub magic: u32,
    pub version: u16,
    pub total_somas: u32,
    pub readout_batch_ticks: u32,
    pub maps: Vec<GxoMapDescriptor>,
    pub pixel_indices: Vec<GxoPixelIndex>,
    pub mapped_soma_ids: Vec<u32>,
}

impl GxoFile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open .gxo file at {:?}", path.as_ref()))?;
            
        let mut magic_buf = [0u8; 4];
        file.read_exact(&mut magic_buf)?;
        let magic = u32::from_le_bytes(magic_buf);
        if magic != 0x47584F30 {
            anyhow::bail!("Invalid GXO magic");
        }

        let mut version_buf = [0u8; 2];
        file.read_exact(&mut version_buf)?;
        let version = u16::from_le_bytes(version_buf);

        let mut num_maps_buf = [0u8; 2];
        file.read_exact(&mut num_maps_buf)?;
        let num_maps = u16::from_le_bytes(num_maps_buf);

        let mut total_somas_buf = [0u8; 4];
        file.read_exact(&mut total_somas_buf)?;
        let total_somas = u32::from_le_bytes(total_somas_buf);

        let mut batch_ticks_buf = [0u8; 4];
        file.read_exact(&mut batch_ticks_buf)?;
        let readout_batch_ticks = u32::from_le_bytes(batch_ticks_buf);

        let mut maps = Vec::with_capacity(num_maps as usize);
        let mut total_pixels = 0;

        for _ in 0..num_maps {
            let mut name_buf = [0u8; 32];
            file.read_exact(&mut name_buf)?;
            let first_null = name_buf.iter().position(|&b| b == 0).unwrap_or(32);
            let name = String::from_utf8_lossy(&name_buf[..first_null]).to_string();

            let mut w_buf = [0u8; 2];
            file.read_exact(&mut w_buf)?;
            let width = u16::from_le_bytes(w_buf);

            let mut h_buf = [0u8; 2];
            file.read_exact(&mut h_buf)?;
            let height = u16::from_le_bytes(h_buf);

            total_pixels += (width as u32) * (height as u32);

            let mut o_buf = [0u8; 4];
            file.read_exact(&mut o_buf)?;
            let somas_offset = u32::from_le_bytes(o_buf);

            let mut c_buf = [0u8; 4];
            file.read_exact(&mut c_buf)?;
            let somas_count = u32::from_le_bytes(c_buf);

            maps.push(GxoMapDescriptor {
                name,
                width,
                height,
                somas_offset,
                somas_count,
            });
        }

        let mut pixel_indices = Vec::with_capacity(total_pixels as usize);
        for _ in 0..total_pixels {
            let mut o_buf = [0u8; 4];
            file.read_exact(&mut o_buf)?;
            let offset = u32::from_le_bytes(o_buf);

            let mut c_buf = [0u8; 2];
            file.read_exact(&mut c_buf)?;
            let count = u16::from_le_bytes(c_buf);

            pixel_indices.push(GxoPixelIndex { offset, count });
        }

        let mut mapped_soma_ids = Vec::with_capacity(total_somas as usize);
        for _ in 0..total_somas {
            let mut id_buf = [0u8; 4];
            file.read_exact(&mut id_buf)?;
            let soma_id = u32::from_le_bytes(id_buf);
            mapped_soma_ids.push(soma_id);
        }

        Ok(Self {
            magic,
            version,
            total_somas,
            readout_batch_ticks,
            maps,
            pixel_indices,
            mapped_soma_ids,
        })
    }
}

use std::path::Path;
use genesis_core::ipc::{GhostsHeader, GhostConnection, GHST_MAGIC};

/// Возвращает (src_soma_ids, target_ghost_ids)
/// Поддерживает два формата файлов .ghosts:
/// 1. GHST (Header + GhostConnection Records)
/// 2. Legacy (u32 Count + flat u32 arrays)
pub fn load_ghosts(path: &Path) -> (Vec<u32>, Vec<u32>) {
    let bytes = std::fs::read(path).expect("Fatal: Failed to read .ghosts file");
    
    if bytes.len() < 4 {
        panic!("Fatal: .ghosts file {:?} is too small ({} bytes)", path, bytes.len());
    }

    let first_u32 = u32::from_le_bytes(bytes[0..4].try_into().unwrap());

    if first_u32 == GHST_MAGIC {
        // --- НОВЫЙ ФОРМАТ (GHST) ---
        if bytes.len() < 16 {
            panic!("Fatal: GHST file {:?} header too small", path);
        }
        unsafe {
            let header_ptr = bytes.as_ptr() as *const GhostsHeader;
            let header = *header_ptr;
            let count = header.connection_count as usize;
            
            let mut src_soma_ids = Vec::with_capacity(count);
            let mut target_ghost_ids = Vec::with_capacity(count);

            let conn_ptr = bytes.as_ptr().add(16) as *const GhostConnection;
            for i in 0..count {
                let conn = *conn_ptr.add(i);
                src_soma_ids.push(conn.src_soma_id);
                target_ghost_ids.push(conn.target_ghost_id);
            }
            (src_soma_ids, target_ghost_ids)
        }
    } else {
        // --- LEGACY ФОРМАТ (u32 count + flat blobs) ---
        let count = first_u32 as usize;
        let expected_size = 4 + (count * 4 * 2);
        if bytes.len() < expected_size {
            panic!("Fatal: Legacy .ghosts file {:?} truncated. Expected {}, got {}", 
                path, expected_size, bytes.len());
        }

        unsafe {
            let src_ptr = bytes.as_ptr().add(4) as *const u32;
            let dst_ptr = bytes.as_ptr().add(4 + count * 4) as *const u32;

            let mut src = Vec::with_capacity(count);
            let mut dst = Vec::with_capacity(count);

            std::ptr::copy_nonoverlapping(src_ptr, src.as_mut_ptr(), count);
            std::ptr::copy_nonoverlapping(dst_ptr, dst.as_mut_ptr(), count);

            src.set_len(count);
            dst.set_len(count);

            (src, dst)
        }
    }
}

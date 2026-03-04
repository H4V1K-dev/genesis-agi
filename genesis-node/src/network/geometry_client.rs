use anyhow::{Context, Result};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};

use super::slow_path::{GeometryRequest, GeometryResponse};

/// Connects to a GeometryServer, sends a request, and waits for a response.
pub async fn send_geometry_request(
    target_addr: SocketAddr,
    req: &GeometryRequest,
) -> Result<GeometryResponse> {
    let mut stream = TcpStream::connect(target_addr).await
        .with_context(|| format!("Failed to connect to GeometryServer at {}", target_addr))?;

    // Serialize payload
    let encoded = bincode::serialize(req).context("Failed to serialize GeometryRequest")?;

    // Send Length-prefix (4 bytes, little-endian)
    let len = encoded.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;

    // Send payload
    stream.write_all(&encoded).await?;

    // Read Length-prefix of response
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await
        .context("Failed to read length prefix from GeometryResponse")?;
    let resp_len = u32::from_le_bytes(len_buf) as usize;

    // Read payload
    let mut resp_buf = vec![0u8; resp_len];
    stream.read_exact(&mut resp_buf).await
        .context("Failed to read payload from GeometryResponse")?;

    let resp: GeometryResponse = bincode::deserialize(&resp_buf)
        .context("Failed to deserialize GeometryResponse")?;

    Ok(resp)
}

pub struct GeometryServer {
    listener: TcpListener,
}

impl GeometryServer {
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        Ok(Self { listener })
    }

    /// Spawns the server loop serving the provided u32 neuron data.
    pub fn spawn(self, geometry_data: Vec<u32>) {
        let num_neurons = geometry_data.len();
        
        let mut buf = Vec::with_capacity(8 + num_neurons * 4);
        buf.extend_from_slice(b"GEOM"); 
        buf.extend_from_slice(&(num_neurons as u32).to_le_bytes());
        
        // Zero-cost cast to bytes
        let data_bytes = bytemuck::cast_slice(&geometry_data);
        buf.extend_from_slice(data_bytes);

        let shared_payload = std::sync::Arc::new(buf);

        tokio::spawn(async move {
            println!("[Geometry Server] Listening on TCP {}", self.listener.local_addr().unwrap());
            loop {
                let (mut stream, _) = match self.listener.accept().await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                
                let data = shared_payload.clone();
                tokio::spawn(async move {
                    // Wait for the "GEOM" request magic
                    let mut magic = [0u8; 4];
                    if let Ok(_) = stream.read_exact(&mut magic).await {
                        if &magic == b"GEOM" {
                            let _ = stream.write_all(&data).await;
                        }
                    }
                });
            }
        });
    }
}

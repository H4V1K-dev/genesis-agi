use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use tokio::net::UdpSocket;
use genesis_core::ipc::{ExternalIoHeader, RouteUpdate, ROUT_MAGIC};
use genesis_core::constants::{GSIO_MAGIC, GSOO_MAGIC, MAX_UDP_PAYLOAD};
use genesis_compute::memory::PinnedBuffer;
use crate::network::router::RoutingTable;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::net::SocketAddr;

/// Контекст I/O для конкретной зоны. Содержит персональный InputSwapchain.
pub struct ZoneIoContext {
    pub swapchain: Arc<InputSwapchain>,
    pub matrix_offsets: HashMap<u32, u32>,
}

/// Lock-Free Double Buffering for External I/O.
/// Isolates the asynchronous Tokio network thread from the synchronous GPU Orchestrator.
pub struct InputSwapchain {
    /// Pointer to the buffer that is currently quiescent and ready for GPU DMA.
    pub ready_for_gpu: AtomicPtr<u8>,
    /// Pointer to the buffer currently being written to by the network thread.
    /// Access is serialized by the single-threaded UDP receive loop.
    back_buffer: AtomicPtr<u8>,
    /// [DOD] Size of allocated Pinned RAM. Hard barrier against network overflow.
    capacity: usize,
    
    // Ownership of the underlying pinned memory
    _buffer_a: PinnedBuffer<u8>,
    _buffer_b: PinnedBuffer<u8>,
}

impl InputSwapchain {
    pub fn new(capacity: usize) -> Result<Self> {
        let buffer_a = PinnedBuffer::new(capacity)?;
        let buffer_b = PinnedBuffer::new(capacity)?;
        
        // Ensure buffers are zeroed initially
        unsafe {
            std::ptr::write_bytes(buffer_a.as_mut_ptr(), 0, capacity);
            std::ptr::write_bytes(buffer_b.as_mut_ptr(), 0, capacity);
        }

        Ok(Self {
            ready_for_gpu: AtomicPtr::new(buffer_a.as_mut_ptr()),
            back_buffer: AtomicPtr::new(buffer_b.as_mut_ptr()),
            capacity,
            _buffer_a: buffer_a,
            _buffer_b: buffer_b,
        })
    }

    /// Writes the payload into the back buffer at a specific byte offset.
    /// Does NOT swap the buffer - caller must call .swap() explicitly (e.g. at the end of a tick).
    pub fn write_incoming_at(&self, offset: usize, payload: &[u8]) {
        let back = self.back_buffer.load(Ordering::Relaxed);
        
        // [DOD] Hard barrier: panic on CPU is 100x better than silent VRAM corruption
        assert!(
            offset + payload.len() <= self.capacity,
            "FATAL DMA BUFFER OVERFLOW: {} bytes at offset {} into buffer of {} bytes",
            payload.len(), offset, self.capacity
        );

        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), back.add(offset), payload.len());
        }
    }

    /// Atomically swaps the back buffer with the ready buffer.
    /// Should be called exactly once per simulation tick/batch by the Orchestrator.
    pub fn swap(&self) {
        let back = self.back_buffer.load(Ordering::Relaxed);
        // Release ordering ensures all previous writes to 'back' are visible to the GPU.
        let old_ready = self.ready_for_gpu.swap(back, Ordering::Release);
        self.back_buffer.store(old_ready, Ordering::Relaxed);
    }

    /// Returns the pointer to the buffer ready for GPU.
    pub fn consume_for_gpu(&self) -> *const u8 {
        self.ready_for_gpu.load(Ordering::Acquire)
    }
}

pub struct ExternalIoServer {
    pub is_sleeping: Arc<AtomicBool>,
    pub oversized_skips: AtomicU64,
    pub dashboard: Arc<crate::tui::DashboardState>,
    pub routing_table: Arc<RoutingTable>,
    pub socket: Arc<UdpSocket>,
    
    /// [DOD] Плоский массив контекстов зон. O(N) линейный поиск оптимален для N < 10.
    pub io_contexts: Vec<(u32, ZoneIoContext)>,
    
    // Validation hashes (Deprecated for multi-zone, use io_contexts)
    // pub zone_hash: u32,
    // pub matrix_hash: u32,

    // R-STDP Dopamine Modulator (Global Reward Broadcast)
    pub global_dopamine: Arc<std::sync::atomic::AtomicI32>,
}

impl ExternalIoServer {
    pub fn new(
        is_sleeping: Arc<AtomicBool>, 
        io_contexts: Vec<(u32, ZoneIoContext)>,
        dashboard: Arc<crate::tui::DashboardState>,
        routing_table: Arc<RoutingTable>,
        socket: Arc<UdpSocket>,
    ) -> Result<Self> {
        Ok(Self {
            is_sleeping,
            oversized_skips: AtomicU64::new(0),
            io_contexts,
            dashboard,
            routing_table,
            socket,
            global_dopamine: Arc::new(std::sync::atomic::AtomicI32::new(0)),
        })
    }

    /// Runs the UDP receiver loop.
    pub async fn run_rx_loop(&self) {
        let mut buf = [0u8; 65536];
        println!("[ExternalIO] UDP Receiver Loop Started on {}", self.socket.local_addr().unwrap());
        loop {
            if let Ok((len, _addr)) = self.socket.recv_from(&mut buf).await {
                self.process_incoming_udp(&buf[..len]);
            }
        }
    }

    /// Processes a raw UDP payload according to Spec §12.
    pub fn process_incoming_udp(&self, payload: &[u8]) {
        self.dashboard.udp_in_packets.fetch_add(1, Ordering::Relaxed);

        // [Contract §12.3] Biological Drop
        if self.is_sleeping.load(Ordering::Acquire) {
            return;
        }

        // [Contract §12.2] EMSGSIZE Protection
        if payload.len() > MAX_UDP_PAYLOAD {
            self.dashboard.oversized_skips.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // [Contract §12.1] Header Validation
        if payload.len() < 4 {
            return;
        }

        let magic = u32::from_le_bytes(payload[0..4].try_into().unwrap());

        // [Step 19] RouteUpdate check
        if magic == ROUT_MAGIC {
            if payload.len() >= std::mem::size_of::<RouteUpdate>() {
                let update = unsafe { &*(payload.as_ptr() as *const RouteUpdate) };
                
                // Nuclear RCU update: Atomic swap of the entire table
                let mut new_map = unsafe { (*self.routing_table.get_map_ptr()).clone() };
                let ipv4 = std::net::Ipv4Addr::from(update.new_ipv4);
                let new_addr = SocketAddr::from((ipv4, update.new_port));
                
                new_map.insert(update.zone_hash, new_addr);
                unsafe { self.routing_table.update_routes(new_map); }
                
                println!("[I/O Server] Route updated for zone 0x{:08X} -> {}", update.zone_hash, new_addr);
            }
            return;
        }

        if payload.len() < std::mem::size_of::<ExternalIoHeader>() {
            return;
        }

        let header = unsafe { &*(payload.as_ptr() as *const ExternalIoHeader) };

        // Magic number check (GSIO for Input)
        if header.magic != GSIO_MAGIC {
            return;
        }

        // [DOD] Zero-Cost Routing (Linear search is optimal for small arrays)
        // Ищем контекст зоны в плоском массиве, который уже в L1
        let ctx = match self.io_contexts.iter().find(|(h, _)| *h == header.zone_hash) {
            Some((_, ctx)) => ctx,
            None => return, // Drop packet for unknown zone
        };

        // Проверка матрицы внутри контекста зоны
        let offset = match ctx.matrix_offsets.get(&header.matrix_hash) {
            Some(&off) => off as usize,
            None => return, // Unknown matrix for this zone
        };

        let payload_start = std::mem::size_of::<ExternalIoHeader>();
        let payload_data = &payload[payload_start..];
        
        if payload_data.len() != header.payload_size as usize {
            return; // Corrupt size field
        }

        // [Contract §12.3] Lock-Free Zero-Copy Transfer (Personalized per Zone)
        ctx.swapchain.write_incoming_at(offset, payload_data);
        // println!("[I/O Server] RX Input for zone 0x{:08X}: {} bytes", header.zone_hash, payload_data.len());

        // Update global dopamine reward for R-STDP
        self.global_dopamine.store(header.global_reward as i32, Ordering::Relaxed);
        if header.global_reward != 0 {
            println!("💉 [Dopamine] Reward Received: {}", header.global_reward);
        }
    }

    /// Отправка Output_History (Вызывается оркестратором после RecordReadout)
    pub async fn send_output_batch(
        &self, 
        target_addr: &str, 
        zone_hash: u32, 
        matrix_hash: u32, 
        pinned_output_addr: usize, 
        output_bytes: usize,
        tx_buffer: &mut [u8] // [DOD] Переиспользуемый буфер от Caller'а
    ) {
        let total_size = std::mem::size_of::<ExternalIoHeader>() + output_bytes;
        if total_size > 65535 || total_size > tx_buffer.len() {
            panic!("Output batch exceeds UDP MTU or buffer capacity.");
        }

        unsafe {
            let header = tx_buffer.as_mut_ptr() as *mut ExternalIoHeader;
            (*header).magic = GSOO_MAGIC; // Contract §12
            (*header).zone_hash = zone_hash;
            (*header).matrix_hash = matrix_hash;
            (*header).payload_size = output_bytes as u32;
            (*header).global_reward = self.global_dopamine.load(Ordering::Relaxed) as i16;
            (*header)._padding = 0;

            std::ptr::copy_nonoverlapping(
                pinned_output_addr as *const u8,
                tx_buffer.as_mut_ptr().add(std::mem::size_of::<ExternalIoHeader>()),
                output_bytes
            );
        }

        let _ = self.socket.send_to(&tx_buffer[..total_size], target_addr).await;
        // println!("[I/O Server] TX Output for zone 0x{:08X}: {} bytes to {}", zone_hash, output_bytes, target_addr);
        
        self.dashboard.udp_out_packets.fetch_add(1, Ordering::Relaxed);
    }

    /// Main loop for the UDP Input Server (Port 8081).
    pub async fn run_input_loop(self: Arc<Self>, addr: &str) -> std::io::Result<()> {
        let socket = UdpSocket::bind(addr).await?;
        println!("[I/O Server] Listening on UDP {}", addr);
        
        let mut buf = vec![0u8; 65536]; // MTU + buffer
        
        loop {
            let (len, _) = socket.recv_from(&mut buf).await?;
            self.process_incoming_udp(&buf[..len]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[tokio::test]
    async fn test_udp_oversized_drop() {
        let is_sleeping = Arc::new(AtomicBool::new(false));
        let dashboard = Arc::new(crate::tui::DashboardState::new(false));
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let server = ExternalIoServer::new(is_sleeping, 1024, 0, 0, dashboard, routing_table, socket).unwrap();
        
        let huge_payload = vec![0u8; MAX_UDP_PAYLOAD + 1];
        server.process_incoming_udp(&huge_payload);
        
        assert_eq!(server.oversized_skips.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_io_multiplexing() {
        let is_sleeping = Arc::new(AtomicBool::new(false));
        let zone_hash = 0x1234;
        let matrix_hash = 0xDEADBEEF;
        let dashboard = Arc::new(crate::tui::DashboardState::new(false));
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let server = ExternalIoServer::new(is_sleeping, 1024, zone_hash, matrix_hash, dashboard, routing_table, socket).unwrap();
        
        let mut packet = Vec::new();
        let header = ExternalIoHeader::new(GSIO_MAGIC, zone_hash, matrix_hash, 4);
        
        unsafe {
            let header_bytes = std::slice::from_raw_parts(
                (&header as *const ExternalIoHeader) as *const u8,
                16
            );
            packet.extend_from_slice(header_bytes);
        }
        packet.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        
        server.process_incoming_udp(&packet);
        
        let ready_ptr = server.swapchain.consume_for_gpu();
        let data = unsafe { std::slice::from_raw_parts(ready_ptr, 4) };
        assert_eq!(data, &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[tokio::test]
    async fn test_route_broadcast_rcu() {
        let is_sleeping = Arc::new(AtomicBool::new(false));
        let dashboard = Arc::new(crate::tui::DashboardState::new(false));
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));
        
        let zone_a = 0xAAAA_AAAA;
        let old_addr: SocketAddr = "1.1.1.1:1111".parse().unwrap();
        unsafe {
            let mut map = HashMap::new();
            map.insert(zone_a, old_addr);
            routing_table.update_routes(map);
        }
        
        assert_eq!(routing_table.get_address(zone_a), Some(old_addr));

        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let server = ExternalIoServer::new(
            is_sleeping, 1024, 0, 0, dashboard, routing_table.clone(), socket
        ).unwrap();

        let new_ip = std::net::Ipv4Addr::new(2, 2, 2, 2);
        let new_port = 2222;
        let update = RouteUpdate {
            magic: ROUT_MAGIC,
            zone_hash: zone_a,
            new_ipv4: u32::from_be_bytes(new_ip.octets()),
            new_port,
            _padding: 0,
        };
        
        let packet = bytemuck::bytes_of(&update);
        server.process_incoming_udp(packet);
        
        let new_addr: SocketAddr = "2.2.2.2:2222".parse().unwrap();
        assert_eq!(routing_table.get_address(zone_a), Some(new_addr));
    }
}

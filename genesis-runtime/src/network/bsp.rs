use std::collections::HashMap;
use std::net::SocketAddr;
use crate::network::SpikeEvent;
use crate::network::ring_buffer::SpikeSchedule;
use crate::network::socket::NodeSocket;

/// The Bulk Synchronous Parallel barrier.
pub struct BspBarrier {
    pub schedule_a: SpikeSchedule,
    pub schedule_b: SpikeSchedule,
    pub writing_to_b: bool,
    pub outgoing_batches: HashMap<u32, Vec<SpikeEvent>>,
    pub socket: Option<NodeSocket>, // None for offline tests
    pub peer_addresses: HashMap<u32, SocketAddr>, // Map target_shard_id -> IP
}

impl BspBarrier {
    pub fn new(sync_batch_ticks: usize) -> Self {
        Self {
            schedule_a: SpikeSchedule::new(sync_batch_ticks),
            schedule_b: SpikeSchedule::new(sync_batch_ticks),
            writing_to_b: true,
            outgoing_batches: HashMap::new(),
            socket: None,
            peer_addresses: HashMap::new(),
        }
    }

    /// Executed by the Orchestrator at the end of the Day Phase batch.
    pub async fn sync_and_swap(&mut self, new_outgoing: HashMap<u32, Vec<SpikeEvent>>, batch_id: u32) -> anyhow::Result<()> {
        self.outgoing_batches = new_outgoing;
        
        if let Some(socket) = &self.socket {
            // 1. Send all outgoing packets concurrently using UDP
            for (target_shard, events) in &self.outgoing_batches {
                if let Some(addr) = self.peer_addresses.get(target_shard) {
                    socket.send_batch(*addr, batch_id, events).await?;
                } else {
                    eprintln!("Warning: No peer address for layout shard {}", target_shard);
                }
            }
            
            // 2. Wait for incoming packets from ALL expected peers
            // (Assuming complete graph for now to unblock. We wait for N-1 peers).
            let expected_packets = self.peer_addresses.len();
            let mut received_events = Vec::new();
            
            for _ in 0..expected_packets {
                let (_, rcv_batch_id, events) = socket.recv_batch().await?;
                if rcv_batch_id != batch_id {
                    // Stale or future batch packet received out of order.
                    eprintln!("Warning: Received batch {} while expecting {}", rcv_batch_id, batch_id);
                } else {
                    received_events.push(events);
                }
            }
            
            // 3. Batch ingest to avoid borrowing self while mutating
            for events in received_events {
                self.ingest_spike_batch(&events);
            }
        }

        self.writing_to_b = !self.writing_to_b;

        // Reset the schedule we are about to start writing into for the next batch
        if self.writing_to_b {
            self.schedule_b.clear();
        } else {
            self.schedule_a.clear();
        }
        
        Ok(())
    }

    /// Ingestion from network socket
    pub fn ingest_spike_batch(&mut self, spikes: &[SpikeEvent]) {
        let schedule = if self.writing_to_b {
            &mut self.schedule_b
        } else {
            &mut self.schedule_a
        };

        for s in spikes {
            let _ = schedule.schedule_spike(s.receiver_ghost_id, s.tick_offset as u32);
        }
    }

    /// Read the schedule for the current Day Phase on the GPU
    pub fn get_active_schedule(&self) -> &SpikeSchedule {
        if self.writing_to_b {
            &self.schedule_a
        } else {
            &self.schedule_b
        }
    }
}

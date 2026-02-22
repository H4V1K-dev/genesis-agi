use std::collections::HashMap;
use crate::network::SpikeEvent;
use crate::network::ring_buffer::SpikeSchedule;

/// The Bulk Synchronous Parallel barrier.
pub struct BspBarrier {
    pub schedule_a: SpikeSchedule,
    pub schedule_b: SpikeSchedule,
    pub writing_to_b: bool,
    pub outgoing_batches: HashMap<u32, Vec<SpikeEvent>>,
}

impl BspBarrier {
    pub fn new(sync_batch_ticks: usize) -> Self {
        Self {
            schedule_a: SpikeSchedule::new(sync_batch_ticks),
            schedule_b: SpikeSchedule::new(sync_batch_ticks),
            writing_to_b: true,
            outgoing_batches: HashMap::new()
        }
    }

    /// Executed by the Orchestrator at the end of the Day Phase batch.
    pub fn sync_and_swap(&mut self, new_outgoing: HashMap<u32, Vec<SpikeEvent>>) {
        // Here we would:
        // 1. Send our outgoing spikes.
        self.outgoing_batches = new_outgoing;
        //    (In a real socket implementation, we wouldn't just store them, we'd transmit them over UDP)
        // 2. Wait for incoming UDP/TCP packets and fill the writing schedule.

        self.writing_to_b = !self.writing_to_b;

        // Reset the schedule we are about to start writing into for the next batch
        if self.writing_to_b {
            self.schedule_b.clear();
        } else {
            self.schedule_a.clear();
        }
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

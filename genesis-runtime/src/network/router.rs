use std::collections::HashMap;
use crate::network::SpikeEvent;

/// Defines a destination for a spike. 
/// A single local neuron might project to multiple Ghost Targets across different nodes.
#[derive(Clone, Debug)]
pub struct GhostTarget {
    pub node_id: u32,
    pub ghost_id: u32,
    pub tick_offset: u8,
}

/// The SpikeRouter maps a local dense neuron ID (extracted from the GPU) 
/// into one or more `SpikeEvent`s destined for remote nodes.
pub struct SpikeRouter {
    /// Maps generic Local Neuron Dense ID -> Array of Ghost Targets
    pub routing_table: HashMap<u32, Vec<GhostTarget>>,
    
    /// The accumulated batches per node, ready for flushing at the end of the Day Phase.
    pub outgoing_spikes: HashMap<u32, Vec<SpikeEvent>>,
}

impl SpikeRouter {
    pub fn new() -> Self {
        Self {
            routing_table: HashMap::new(),
            outgoing_spikes: HashMap::new(),
        }
    }

    /// Add a manual subscription mapping (useful for testing and slow-path geography setup).
    pub fn add_route(&mut self, local_id: u32, target: GhostTarget) {
        self.routing_table.entry(local_id).or_default().push(target);
    }

    /// Called natively per-tick by the Day Phase orchestration.
    pub fn route_spikes(&mut self, local_spikes: &[u32], current_tick_offset: u32) {
        for &nid in local_spikes {
            if let Some(targets) = self.routing_table.get(&nid) {
                // Fan-out: One neuron might send spikes to multiple remote locations
                for t in targets {
                    let total_offset = (current_tick_offset as u32 + t.tick_offset as u32);
                    // We must clamp the offset mapping to u8 (assuming batch_size < 255)
                    let final_offset = std::cmp::min(total_offset, 255) as u8;

                    let event = SpikeEvent {
                        receiver_ghost_id: t.ghost_id,
                        tick_offset: final_offset,
                        _pad: [0; 3],
                    };

                    self.outgoing_spikes.entry(t.node_id).or_default().push(event);
                }
            }
        }
    }

    /// Fetches and clears the finalized outgoing buffers. 
    /// Intended for use primarily by the BspBarrier at the end of a Day Batch.
    pub fn flush_outgoing(&mut self) -> HashMap<u32, Vec<SpikeEvent>> {
        let current = std::mem::take(&mut self.outgoing_spikes);
        self.outgoing_spikes = HashMap::new(); // Ensure re-initialization
        current
    }
}

#[cfg(test)]
mod tests {
    use crate::bake::axon_growth::{grow_axons, GrownAxon, LayerZRange, ShardBounds};
    use crate::bake::dendrite_connect::connect_dendrites;
    use crate::bake::layout::ShardStateSoA;
    use crate::bake::neuron_placement::PlacedNeuron;
    use crate::parser::blueprints::NeuronType;
    use crate::parser::simulation::{SimulationConfig, SimulationParams, WorldConfig};
    use genesis_core::coords::{pack_position, unpack_target};

    fn make_sim_config(w: u32, d: u32, h: u32) -> SimulationConfig {
        SimulationConfig {
            world: WorldConfig {
                width_um: w * 50,
                depth_um: d * 50,
                height_um: h * 50,
            },
            simulation: SimulationParams {
                voxel_size_um: 50,
                segment_length_voxels: 1,
                axon_growth_max_steps: 100,
                tick_duration_us: 1000,
                total_ticks: 100_000,
                master_seed: "0".to_string(),
                global_density: 1.0,
                signal_speed_um_tick: 50,
                sync_batch_ticks: 10,
                num_virtual_axons: None,
                night_interval_ticks: 1000,
            },
        }
    }

    fn make_neuron(x: u32, y: u32, z: u32, t: usize) -> PlacedNeuron {
        PlacedNeuron {
            position: pack_position(x, y, z, t as u32),
            type_idx: t,
            layer_name: "TestLayer".to_string(),
        }
    }

    fn make_type(name: &str, is_inhibitory: bool, whitelist: Vec<&str>) -> NeuronType {
        let mut wl = Vec::new();
        for w in whitelist {
            wl.push(w.to_string());
        }

        NeuronType {
            name: name.to_string(),
            growth_vertical_bias: 0.0,
            steering_fov_deg: 90.0,
            steering_radius_um: 150.0,
            steering_weight_inertia: 0.1,
            steering_weight_sensor: 0.9,
            steering_weight_jitter: 0.0,
            initial_synapse_weight: 74,
            dendrite_whitelist: wl,
            is_inhibitory,
            sprouting_weight_type: 0.0,
            type_affinity: 1.0,
            inertia_curve: [0; 16],
            ltm_slot_count: 80,
            conduction_velocity: 1,
            signal_propagation_length: 5,
            homeostasis_penalty: 1,
            homeostasis_decay: 1,
            slot_decay_ltm: 1,
            slot_decay_wm: 1,
            threshold: 1000,
            rest_potential: 0,
            leak_rate: 10,
            refractory_period: 5,
            synapse_refractory_period: 5,
            axon_growth_step: 10,
            gsop_depression: 10,
            gsop_potentiation: 10,
            sprouting_weight_distance: 1.0, // Used in scoring
            sprouting_weight_power: 0.0,
            sprouting_weight_explore: 0.0,
            prune_threshold: 10,
        }
    }

    fn make_axon(soma_idx: usize, type_idx: usize, segments: Vec<u32>) -> GrownAxon {
        // Mock a simple grown axon where tip is the last segment
        let last = *segments.last().unwrap_or(&0);
        let tip_z = (last >> 20) & 0xFF;
        let tip_y = (last >> 10) & 0x3FF;
        let tip_x = last & 0x3FF;
        
        GrownAxon {
            soma_idx,
            type_idx,
            tip_x,
            tip_y,
            tip_z,
            length_segments: segments.len() as u32,
            segments,
        }
    }

    fn pack_seg(x: u32, y: u32, z: u32, t: u32) -> u32 {
        (t << 28) | (z << 20) | (y << 10) | x
    }

    #[test]
    fn test_basic_connection() {
        let neurons = vec![
            make_neuron(0, 0, 0, 0), // A
            make_neuron(10, 10, 0, 0), // B
        ];
        let types = vec![make_type("Type0", false, vec![])];
        
        // Axon from A passes near B
        let a_ax = make_axon(0, 0, vec![
            pack_seg(0, 0, 0, 0),
            pack_seg(5, 5, 0, 0),
            pack_seg(10, 9, 0, 0), // Very close to B (10, 10, 0)
            pack_seg(15, 15, 0, 0),
        ]);
        
        let mut shard = ShardStateSoA::new_blank(2, 1, 0);
        connect_dendrites(&mut shard, &neurons, &[a_ax], &types, 42);

        let iter_range = 0..genesis_core::constants::MAX_DENDRITE_SLOTS;
        let p_n = shard.padded_n;

        // B should connect to A's axon
        let mut b_connected = false;
        for slot in iter_range.clone() {
            let target = shard.dendrite_targets[slot * p_n + 1];
            if target != 0 {
                b_connected = true;
                let weight = shard.dendrite_weights[slot * p_n + 1];
                let expected_w = types[0].initial_synapse_weight as i16;
                assert_eq!(weight, expected_w);
            }
        }
        assert!(b_connected, "Neuron B did not connect to Neuron A's axon");

        // A should NOT connect to its own axon (self_exclusion)
        let mut a_connected = false;
        for slot in iter_range {
            let target = shard.dendrite_targets[slot * p_n + 0];
            if target != 0 {
                a_connected = true;
            }
        }
        assert!(!a_connected, "Neuron A connected to its own axon");
    }

    #[test]
    fn test_rule_of_uniqueness() {
        let neurons = vec![
            make_neuron(0, 0, 0, 0), // A
            make_neuron(10, 10, 0, 0), // B
        ];
        let types = vec![make_type("Type0", false, vec![])];
        
        // Axon from A passes near B and circles around it. Add a dummy first segment.
        let a_ax = make_axon(0, 0, vec![
            pack_seg(0, 0, 0, 0), // seg 0 (so closest is not seg 0)
            pack_seg(10, 9, 0, 0),
            pack_seg(9, 10, 0, 0),
            pack_seg(10, 11, 0, 0),
            pack_seg(11, 10, 0, 0),
        ]);
        
        let mut shard = ShardStateSoA::new_blank(2, 1, 0);
        connect_dendrites(&mut shard, &neurons, &[a_ax], &types, 42);

        let mut connections_count = 0;
        let p_n = shard.padded_n;
        for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
            if shard.dendrite_targets[slot * p_n + 1] != 0 {
                connections_count += 1;
            }
        }
        // Even though multiple segments are close, only ONE connection should form
        assert_eq!(connections_count, 1);
    }

    #[test]
    fn test_whitelist_filter() {
        let neurons = vec![
            make_neuron(0, 0, 0, 0), // soma: E
            make_neuron(10, 10, 0, 1), // I
        ];
        let types = vec![
            make_type("E", false, vec!["I"]), // Only connects to I
            make_type("I", true, vec![]),
        ];
        
        // Axon from E
        let ax_e = make_axon(0, 0, vec![pack_seg(0, 0, 0, 0), pack_seg(10, 10, 0, 0)]); // Same pos as I
        // Axon from I
        let ax_i = make_axon(1, 1, vec![pack_seg(0, 0, 0, 1), pack_seg(0, 0, 0, 1)]); // Same pos as E

        let mut shard = ShardStateSoA::new_blank(2, 2, 0);
        connect_dendrites(&mut shard, &neurons, &[ax_e, ax_i], &types, 42);
        
        let p_n = shard.padded_n;
        
        // E should only connect to I's axon
        let mut e_connected_to_i = false;
        for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
            let target = shard.dendrite_targets[slot * p_n + 0];
            if target != 0 {
                let (ax_id, _) = unpack_target(target).unwrap();
                assert_eq!(ax_id, 1, "E connected to something other than I");
                e_connected_to_i = true;
            }
        }
        assert!(e_connected_to_i);

        // I connects to everyone (whitelist empty), so I connects to E's axon
        let mut i_connected_to_e = false;
        for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
            let target = shard.dendrite_targets[slot * p_n + 1];
            if target != 0 {
                let (ax_id, _) = unpack_target(target).unwrap();
                assert_eq!(ax_id, 0); // E's axon is idx 0
                i_connected_to_e = true;
            }
        }
        assert!(i_connected_to_e);
    }

    #[test]
    fn test_inhibitory_weight_sign() {
        let neurons = vec![
            make_neuron(10, 10, 0, 0), // Target
        ];
        let types = vec![
            make_type("Target", false, vec![]),
            make_type("Inhibitory", true, vec![]),
        ];
        
        let mut ax_i = make_axon(1, 1, vec![pack_seg(0,0,0,1), pack_seg(10, 10, 0, 1)]);
        ax_i.soma_idx = 1; // Different soma

        let mut shard = ShardStateSoA::new_blank(1, 2, 0);
        connect_dendrites(&mut shard, &neurons, &[ax_i], &types, 42);

        let p_n = shard.padded_n;
        let w = shard.dendrite_weights[0 * p_n + 0];
        
        assert!(w < 0, "Inhibitory synapse should have negative weight, got {}", w);
        assert_eq!(w, -74);
    }

    #[test]
    fn test_distant_axon_ignored() {
        let neurons = vec![make_neuron(0, 0, 0, 0)];
        let types = vec![make_type("T", false, vec![])];
        
        let ax = make_axon(1, 0, vec![pack_seg(100, 100, 100, 0)]); // Far away

        let mut shard = ShardStateSoA::new_blank(1, 2, 0);
        connect_dendrites(&mut shard, &neurons, &[ax], &types, 42);

        let p_n = shard.padded_n;
        for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
            assert_eq!(shard.dendrite_targets[slot * p_n + 0], 0);
        }
    }

    #[test]
    fn test_multiple_candidates_sorted() {
        let neurons = vec![make_neuron(5, 5, 0, 0)];
        let types = vec![make_type("T", false, vec![])];
        
        let ax1 = make_axon(1, 0, vec![pack_seg(0,0,0,0), pack_seg(5, 6, 0, 0)]); // dist = 1
        let ax2 = make_axon(2, 0, vec![pack_seg(0,0,0,0), pack_seg(5, 8, 0, 0)]); // dist = 3
        let ax3 = make_axon(3, 0, vec![pack_seg(0,0,0,0), pack_seg(5, 10, 0, 0)]); // dist = 5

        let mut shard = ShardStateSoA::new_blank(1, 4, 0);
        // Note: distance scoring makes closer axons score higher!
        connect_dendrites(&mut shard, &neurons, &[ax1, ax2, ax3], &types, 42);

        let p_n = shard.padded_n;
        // Top score in slot 0 should be ax1 (ID 0)
        let (t0, _) = unpack_target(shard.dendrite_targets[0 * p_n + 0]).unwrap_or((999, 999));
        let (t1, _) = unpack_target(shard.dendrite_targets[1 * p_n + 0]).unwrap_or((999, 999));
        let (t2, _) = unpack_target(shard.dendrite_targets[2 * p_n + 0]).unwrap_or((999, 999));
        
        assert_eq!(t0, 0, "Closest axon should be in slot 0");
        assert_eq!(t1, 1, "Next closest in slot 1");
        assert_eq!(t2, 2, "Farthest in slot 2");
    }

    #[test]
    fn test_empty_world() {
        let neurons = vec![];
        let types = vec![make_type("T", false, vec![])];
        let mut shard = ShardStateSoA::new_blank(0, 0, 0);
        
        connect_dendrites(&mut shard, &neurons, &[], &types, 42);
        // Should not panic
    }

    #[test]
    #[ignore]
    fn test_visualize_connectivity() {
        let sim = make_sim_config(40, 40, 10);
        let layers = vec![
            LayerZRange { name: "L1".to_string(), z_start_vox: 0, z_end_vox: 5 },
            LayerZRange { name: "L2".to_string(), z_start_vox: 5, z_end_vox: 10 },
        ];
        let mut types = vec![
            make_type("E_Vert", false, vec![]),
            make_type("I_Horiz", true, vec![]),
        ];
        types[0].growth_vertical_bias = 0.8;
        types[0].axon_growth_step = 20;
        types[1].growth_vertical_bias = 0.0;
        types[1].axon_growth_step = 20;
        types[1].steering_weight_jitter = 0.8; // Wavy
        
        let bounds = ShardBounds::full_world(&sim);

        let mut neurons = vec![];
        let mut rng_seed = 123;
        for i in 0..15 {
            let x = (rng_seed * 7) % 36 + 2;
            rng_seed = (rng_seed * 13) % 1000;
            let y = (rng_seed * 11) % 36 + 2;
            rng_seed = (rng_seed * 17) % 1000;
            
            let t = if i % 3 == 0 { 1 } else { 0 };
            neurons.push(make_neuron(x, y, 5, t));
        }

        let (axons, _ghosts) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);
        
        let mut shard = ShardStateSoA::new_blank(neurons.len(), axons.len(), 0);
        connect_dendrites(&mut shard, &neurons, &axons, &types, 42);

        // Drawing ASCII map for Z=5 plane
        let mut grid_chars = vec![vec![' '; 40]; 40];
        
        // 1. Draw axons
        for ax in &axons {
            // We'll draw segments that are near Z=5
            for (i, &seg) in ax.segments.iter().enumerate() {
                let z = (seg >> 20) & 0xFF;
                let y = (seg >> 10) & 0x3FF;
                let x = seg & 0x3FF;
                
                if z >= 3 && z <= 7 {
                    if i > 0 {
                        // Drawing path roughly
                        if x < 40 && y < 40 {
                            let mut ch = '.';
                            if x == ax.tip_x && y == ax.tip_y {
                                ch = '*';
                            }
                            grid_chars[y as usize][x as usize] = ch;
                        }
                    }
                }
            }
        }

        // 2. Draw Connections (Dendrites)
        let p_n = shard.padded_n;
        let mut conn_count = 0;
        for (i, n) in neurons.iter().enumerate() {
            for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
                let target = shard.dendrite_targets[slot * p_n + i];
                if target != 0 {
                    let (ax_id, _) = unpack_target(target).unwrap();
                    let ax = &axons[ax_id as usize];
                    let weight = shard.dendrite_weights[slot * p_n + i];
                    
                    // Simple drawing: arrow from Soma to Axon tip
                    let tx = ax.tip_x.min(39);
                    let ty = ax.tip_y.min(39);
                    let sx = n.x().min(39);
                    let sy = n.y().min(39);
                    
                    let dx = (tx as i32 - sx as i32).signum();
                    let dy = (ty as i32 - sy as i32).signum();
                    
                    let draw_x = (sx as i32 + dx).clamped(0, 39) as usize;
                    let draw_y = (sy as i32 + dy).clamped(0, 39) as usize;
                    
                    let arrow = if weight > 0 { '+' } else { '-' };
                    grid_chars[draw_y][draw_x] = arrow;
                    conn_count += 1;
                }
            }
        }

        // 3. Draw Somas
        for (_i, n) in neurons.iter().enumerate() {
            let x = n.x().min(39) as usize;
            let y = n.y().min(39) as usize;
            let ch = if n.type_idx == 0 { 'E' } else { 'I' };
            grid_chars[y][x] = ch;
        }

        // Print Map
        println!("\n=== Dendrite Connectivity Map (Z=5, seed=42) ===\n");
        println!("     0123456789012345678901234567890123456789");
        for y in 0..40 {
            print!("{:>2} [ ", y);
            for x in 0..40 {
                print!("{}", grid_chars[y][x]);
            }
            println!(" ]");
        }
        println!("\nLegend: E=Excitatory I=Inhibitory .=Axon path *=Axon Tip +=Exc.Syn -=Inh.Syn");
        println!("Neurons: {}  Axons: {}  Connections: {}", neurons.len(), axons.len(), conn_count);
        println!("================================================\n");
    }
    
    trait ClampedExt {
        fn clamped(self, min: Self, max: Self) -> Self;
    }
    impl ClampedExt for i32 {
        fn clamped(self, min: Self, max: Self) -> Self {
            if self < min { min } else if self > max { max } else { self }
        }
    }
}

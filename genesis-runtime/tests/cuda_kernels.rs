#[path = "mock.rs"]
pub mod mock;

use mock::MockBakerBuilder;
use genesis_runtime::{Runtime, VariantParameters, GenesisConstantMemory};
use genesis_runtime::memory::VramState;

fn setup_constants() -> GenesisConstantMemory {
    let mut constants = GenesisConstantMemory::default();
    // Default variant 0
    constants.variants[0] = VariantParameters {
        threshold: 100,
        rest_potential: 0,
        leak: 2,
        homeostasis_penalty: 10,
        homeostasis_decay: 1,
        gsop_potentiation: 100,
        gsop_depression: 4000,
        refractory_period: 2,
        synapse_refractory: 5,
        slot_decay_ltm: 10,
        slot_decay_wm: 2,
        _padding: [0; 4],
    };
    // Inertia LUT (example: higher weights -> less inertia, simplified)
    for i in 0..16 {
        constants.inertia_lut[i] = (16 - i) as u8;
    }
    constants
}

#[test]
fn test_propagate_axons() {
    let consts = setup_constants();
    Runtime::init_constants(&consts);

    let mut builder = MockBakerBuilder::new(1, 2);
    builder.axon_heads[0] = 50; // Active axon
    // builder.axon_heads[1] is 0x80000000 by default (sentinel)

    let (state_bytes, axons_bytes) = builder.build();
    let vram = VramState::load_shard(&state_bytes, &axons_bytes).unwrap();
    let mut runtime = Runtime::new(vram, 3); // v_seg = 3

    runtime.tick();
    runtime.synchronize();

    let axon_heads = runtime.vram.download_axon_head_index().unwrap();

    // Verify propagation
    assert_eq!(axon_heads[0], 53, "Active axon should advance by v_seg (3)");
    assert_eq!(axon_heads[1], 0x80000000, "Sentinel axon must not advance");
}

#[test]
fn test_update_neurons() {
    let consts = setup_constants();
    Runtime::init_constants(&consts);

    let mut builder = MockBakerBuilder::new(2, 1);
    
    // Neuron 0 setup: should leak
    builder.voltages[0] = 50;
    
    // Neuron 1 setup: exact hit from axon 0 on segment 10, weight 60
    builder.voltages[1] = 45; // Below threshold (100)
    // builder.flags[1] is 0 (variant 0)
    
    // Assuming v_seg = 1, in propagate it will become 10. So we need segment 10.
    builder.axon_heads[0] = 9; // Propagates to 10
    
    // Set dendrite slot 0 for neuron 1
    // builder.set_dendrite(nid, slot, axon_id, segment, weight)
    builder.set_dendrite(1, 0, 0, 10, 60);

    let (state_bytes, axons_bytes) = builder.build();
    let vram = VramState::load_shard(&state_bytes, &axons_bytes).unwrap();
    let mut runtime = Runtime::new(vram, 1);

    runtime.tick();
    runtime.synchronize();

    let voltages = runtime.vram.download_voltage().unwrap();
    let flags = runtime.vram.download_flags().unwrap();

    // Neuron 0 only leaked by 2
    assert_eq!(voltages[0], 48, "Neuron 0 should have leaked 2 voltage");

    // Neuron 1 received 60 dendrite sum, starting from 45. Peak 105. Spikes!
    assert_eq!(voltages[1], consts.variants[0].rest_potential, "Neuron 1 should have reset to rest_potential after spiking");
    assert_eq!(flags[1] & 1, 1, "Neuron 1 should have the spiked flag set");
}

#[test]
fn test_apply_gsop() {
    let consts = setup_constants();
    Runtime::init_constants(&consts);

    // Testing plasticity:
    // Neuron 0 spikes, dendrite weight should increase if timer > 0 (potentiation)
    // Neuron 1 spikes, dendrite weight should decrease if timer == 0 (depression)
    
    let mut builder = MockBakerBuilder::new(2, 2);
    builder.voltages[0] = 200; // Will definitely spike
    builder.voltages[1] = 200; // Will definitely spike

    // Both get target assigned on slot 0
    builder.set_dendrite(0, 0, 0, 10, 100);
    builder.set_dendrite(1, 0, 1, 10, 100);

    // Neuron 0's dendrite has a timer > 0
    builder.dendrite_timers[0] = 3;
    
    // Neuron 1's dendrite timer == 0
    builder.dendrite_timers[1] = 0;

    let (state_bytes, axons_bytes) = builder.build();
    let vram = VramState::load_shard(&state_bytes, &axons_bytes).unwrap();
    let mut runtime = Runtime::new(vram, 1);

    runtime.tick();
    runtime.synchronize();

    let weights = runtime.vram.download_dendrite_weights().unwrap();
    // weights[0] is slot 0 nid 0
    // weights[1] is slot 0 nid 1
    
    let new_w0 = weights[0];
    let new_w1 = weights[1];

    assert!(new_w0 > 100, "Weight 0 should be potentiated, was {} expected > 100", new_w0);
    assert!(new_w1 < 100, "Weight 1 should be depressed, was {} expected < 100", new_w1);
}

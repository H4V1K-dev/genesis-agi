use tokio::runtime::Runtime;
use genesis_runtime::network::socket::NodeSocket;
use genesis_runtime::network::SpikeEvent;

#[test]
fn test_udp_fast_path() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        // Create sender and receiver on local loopback with random ports
        let sender = NodeSocket::bind("127.0.0.1:0").await.unwrap();
        let receiver = NodeSocket::bind("127.0.0.1:0").await.unwrap();
        
        let target_addr = receiver.local_addr().unwrap();
        let batch_id = 42;

        // Generate 1000 mockup spikes
        let mut spikes = Vec::new();
        for i in 0..1000 {
            spikes.push(SpikeEvent {
                receiver_ghost_id: i * 5,
                tick_offset: (i % 20) as u8,
                _pad: [0; 3],
            });
        }

        // Send
        sender.send_batch(target_addr, batch_id, &spikes).await.unwrap();

        // Receive
        let (src, rcv_batch_id, received_spikes) = receiver.recv_batch().await.unwrap();

        assert_eq!(src.port(), sender.local_addr().unwrap().port());
        assert_eq!(rcv_batch_id, batch_id);
        assert_eq!(received_spikes.len(), 1000);

        for i in 0..1000 {
            assert_eq!(received_spikes[i].receiver_ghost_id, i as u32 * 5);
            assert_eq!(received_spikes[i].tick_offset, (i % 20) as u8);
        }
    });
}

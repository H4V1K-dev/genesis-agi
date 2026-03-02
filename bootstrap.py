#!/usr/bin/env python3
"""
Bootstrap script to unlock the Genesis distributed simulation.
Sends initial frame to Node A to trigger simulation in both nodes.
"""
import socket
import struct
import time

def fnv1a_32(data: bytes) -> int:
    hash_value = 0x811c9dc5
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value

# Configuration
SENSORY_IP = "127.0.0.1"
SENSORY_PORT = 8081  # External input port for SensoryCortex

# Hash values
SENSORY_ZONE_HASH = fnv1a_32(b"SensoryCortex")
SENSORY_MATRIX_HASH = fnv1a_32(b"sensory_in")

# Create socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"[Bootstrap] Sending initial frame to {SENSORY_IP}:{SENSORY_PORT}")
print(f"[Bootstrap] Zone Hash: 0x{SENSORY_ZONE_HASH:08x}")
print(f"[Bootstrap] Matrix Hash: 0x{SENSORY_MATRIX_HASH:08x}")

# Send 100 frames of zeros (one batch cycle)
for frame_idx in range(100):
    # ExternalIoHeader: zone_hash (u32), matrix_hash (u32), payload_size (u32), global_reward (i16), padding (u16)
    # Payload: 256 input bits = 8 words (32-bit each) = 32 bytes
    
    payload = b'\x00' * 32  # 256 bits of zeros
    
    header = struct.pack(
        "<IIIHH",
        SENSORY_ZONE_HASH,      # zone_hash
        SENSORY_MATRIX_HASH,    # matrix_hash
        32,                     # payload_size (32 bytes)
        0,                      # global_reward
        0                       # padding
    )
    
    packet = header + payload
    sock.sendto(packet, (SENSORY_IP, SENSORY_PORT))
    
    if frame_idx % 10 == 0:
        print(f"[Bootstrap] Frame {frame_idx}/100")
    
    time.sleep(0.001)  # Small delay to allow processing

sock.close()
print("[Bootstrap] ✓ Bootstrap complete. Simulation should now be running.")

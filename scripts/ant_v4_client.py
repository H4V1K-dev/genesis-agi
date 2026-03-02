import socket
import struct
import gymnasium as gym
import numpy as np
import threading
import time

def fnv1a_32(data: bytes) -> int:
    hash_value = 0x811c9dc5
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value

ZONE_HASH = fnv1a_32(b"SensoryCortex")
MATRIX_PROP_HASH = fnv1a_32(b"proprioception_joints") 
MATRIX_VEST_HASH = fnv1a_32(b"vestibular_gyro")
MATRIX_TACT_HASH = fnv1a_32(b"tactile_feet")

MOTOR_ZONE_HASH = fnv1a_32(b"MotorCortex")
MOTOR_MATRIX_HASH = fnv1a_32(b"motor_actuators")

GENESIS_IP = "127.0.0.1"
PORT_OUT = 8014 # Node A IO Port
PORT_IN = 8082

class State:
    def __init__(self):
        self.obs = np.zeros(27) 
        self.action = np.zeros(8)
        self.running = True
        self.reset_needed = False
        self.steps_no_progress = 0
        self.last_x = 0.0

state = State()

def encode_population(value, min_val, max_val, neurons=16):
    norm = np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    center_idx = int(norm * (neurons - 1))
    bitmask = 0
    # Gaussian-like spread: center + neighbors
    for i in range(max(0, center_idx - 1), min(neurons, center_idx + 2)):
        bitmask |= (1 << i)
    return bitmask

def udp_hot_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT_IN))
    sock.settimeout(0.01)
    
    while state.running:
        obs = state.obs
        
        # 1. Proprioception (16 joint pos/vel -> 16 populations of 16 neurons)
        prop_words = []
        for i in range(0, 16, 2):
            m1 = encode_population(obs[i+11], -1.2, 1.2, neurons=16) 
            m2 = encode_population(obs[i+12], -1.2, 1.2, neurons=16) 
            prop_words.append(m1 | (m2 << 16))
        
        # Calculate instant dopamine reward based on forward speed (obs[13] contains velocity in X roughly, or we can use diff in state.last_x)
        current_x = state.obs[0]
        dx = current_x - state.last_x
        # Base scale: 0.05 step usually means good speed. Map to [-1024, 1024].
        dopamine = int(np.clip(dx * 10000.0, -1024, 1024))
        
        prop_payload = struct.pack(f"<IIIhH{len(prop_words)}I", ZONE_HASH, MATRIX_PROP_HASH, len(prop_words)*4, dopamine, 0, *prop_words)
        sock.sendto(prop_payload, (GENESIS_IP, PORT_OUT))

        # 2. Vestibular (8 gyro values -> 8 populations of 8 neurons)
        vest_words = []
        for i in range(0, 8, 4):
            v = []
            for j in range(4):
                v.append(encode_population(obs[i+j+3], -2.5, 2.5, neurons=8))
            vest_words.append(v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24))
            
        vest_payload = struct.pack(f"<IIIhH{len(vest_words)}I", ZONE_HASH, MATRIX_VEST_HASH, len(vest_words)*4, dopamine, 0, *vest_words)
        sock.sendto(vest_payload, (GENESIS_IP, PORT_OUT))

        # 3. Tactile (4 feet -> 4 neurons)
        tact_mask = 0
        for i in range(4):
            if obs[i] > 0.0: 
                tact_mask |= (1 << i)
        
        tact_payload = struct.pack(f"<IIIhHI", ZONE_HASH, MATRIX_TACT_HASH, 4, dopamine, 0, tact_mask)
        sock.sendto(tact_payload, (GENESIS_IP, PORT_OUT))
        
        # --- DECODING ANTAGONISTIC MUSCLES ---
        try:
            data, _ = sock.recvfrom(65535)
            # data: header(12) + history(payload_size)
            header = struct.unpack("<III", data[:12])
            if header[0] == MOTOR_ZONE_HASH and header[1] == MOTOR_MATRIX_HASH:
                payload_bytes = data[12:]
                
                # Convert bits to list of 0/1 spikes
                spikes = []
                for b in payload_bytes:
                    for bit in range(8):
                        spikes.append((b >> bit) & 1)
                
                if len(spikes) >= 256:
                    for i in range(8):
                        # Flexor population: (i*2)*16 to (i*2+1)*16
                        flexor_spikes = sum(spikes[(i*2)*16 : (i*2+1)*16])
                        # Extensor population: (i*2+1)*16 to (i*2+2)*16
                        extensor_spikes = sum(spikes[(i*2+1)*16 : (i*2+2)*16])
                        
                        # Force = (Flexor - Extensor) / 16.0
                        state.action[i] = (flexor_spikes - extensor_spikes) / 16.0
                    
        except socket.timeout:
            pass
        except Exception as e:
            pass

def main():
    try:
        # We need exclude_current_positions_from_observation=False to get X, Y for progress tracking
        env = gym.make('Ant-v4', render_mode="human", exclude_current_positions_from_observation=False)
    except:
        env = gym.make('Ant-v4', render_mode="rgb_array", exclude_current_positions_from_observation=False)
        
    state.obs, _ = env.reset()
    state.last_x = state.obs[0] # Now it's the real X coordinate
    
    udp_thread = threading.Thread(target=udp_hot_loop)
    udp_thread.start()
    
    try:
        while True:
            # Step environment
            state.obs, reward, terminated, truncated, info = env.step(state.action)
            
            # --- ARTIFICIAL PAIN (Adaptive Reset) ---
            # In Ant-v4 (non-excluded), index 0 is X, 1 is Y, 2 is Z
            current_x = state.obs[0]
            if current_x > state.last_x + 0.005: # Threshold for progress
                state.steps_no_progress = 0
            else:
                state.steps_no_progress += 1
            state.last_x = current_x
            
            # Reset if flipped over (Z-position < 0.25) or stuck for > 200 steps
            flipped = state.obs[2] < 0.25 
            stuck = state.steps_no_progress > 200
            
            if terminated or truncated or flipped or stuck:
                if flipped or stuck:
                    print(f"Artificial Pain Triggered: {'Flipped' if flipped else 'Stuck'}")
                state.obs, _ = env.reset()
                state.steps_no_progress = 0
                state.last_x = state.obs[0]
            
    except KeyboardInterrupt:
        state.running = False
        udp_thread.join()
        env.close()

if __name__ == "__main__":
    main()

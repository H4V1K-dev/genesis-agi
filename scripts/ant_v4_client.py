import socket
import struct
import random
import gymnasium as gym
import numpy as np
import threading
import time

GSIO_MAGIC = 0x4F495347 # "GSIO"
GSOO_MAGIC = 0x4F4F5347 # "GSOO"

def fnv1a_32(data: bytes) -> int:
    hash_value = 0x811c9dc5
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value

ZONE_HASH = fnv1a_32(b"SensoryCortex")
# Используем хэш ПЕРВОЙ матрицы. Сервер положит весь блоб начиная с offset=0.
MATRIX_PROP_HASH = fnv1a_32(b"proprioception_joints")

MOTOR_ZONE_HASH = fnv1a_32(b"MotorCortex")
MOTOR_MATRIX_HASH = fnv1a_32(b"motor_actuators")

GENESIS_IP = "127.0.0.1"
PORT_OUT = 8081
PORT_IN = 8082

class State:
    def __init__(self):
        self.obs = np.zeros(27)
        self.action = np.zeros(8)
        self.running = True
        self.steps_no_progress = 0
        self.last_x = 0.0

state = State()

def encode_population(value, min_val, max_val, neurons=16):
    norm = np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    center_idx = int(norm * (neurons - 1))
    bitmask = 0
    for i in range(max(0, center_idx - 1), min(neurons, center_idx + 2)):
        bitmask |= (1 << i)
    return bitmask

BATCH_TICKS = 100
# 324 виртуальных аксона / 32 = 10.125 -> 11 слов (u32) на один тик.
WORDS_PER_TICK = 11 

def udp_hot_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT_IN))
    sock.settimeout(0.001)
    print(f"💉 [Client] UDP Loop Started (Batch Size: {BATCH_TICKS})")

    while state.running:
        # [DOD] Идеально выровненная матрица [Тики x Слова]
        batch_bitmask = np.zeros((BATCH_TICKS, WORDS_PER_TICK), dtype=np.uint32)
        total_dopamine = 0

        for t in range(BATCH_TICKS):
            obs = state.obs
            
            # 1. Proprioception (Слова 0..7)
            for i in range(0, 16, 2):
                m1 = encode_population(obs[i+11], -1.2, 1.2, neurons=16)
                m2 = encode_population(obs[i+12], -1.2, 1.2, neurons=16)
                batch_bitmask[t, i//2] = m1 | (m2 << 16)

            # 2. Vestibular (Слова 8..9)
            for i in range(0, 8, 4):
                v0 = encode_population(obs[i+3], -2.5, 2.5, neurons=8)
                v1 = encode_population(obs[i+4], -2.5, 2.5, neurons=8)
                v2 = encode_population(obs[i+5], -2.5, 2.5, neurons=8)
                v3 = encode_population(obs[i+6], -2.5, 2.5, neurons=8)
                batch_bitmask[t, 8 + i//4] = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24)

            # 3. Tactile (Слово 10)
            tact_mask = 0
            for i in range(4):
                if obs[i] > 0.0:
                    tact_mask |= (1 << i)
            batch_bitmask[t, 10] = tact_mask

            current_x = state.obs[0]
            dx = current_x - state.last_x
            total_dopamine += int(np.clip(dx * 10000.0, -1024, 1024))
            state.last_x = current_x

            time.sleep(0.0001)

        avg_dopamine = total_dopamine // BATCH_TICKS

        # Отправляем ЕДИНЫЙ монолитный блоб. Движок распакует его без копирований.
        payload = batch_bitmask.tobytes()
        packet = struct.pack(f"<IIIIhH", GSIO_MAGIC, ZONE_HASH, MATRIX_PROP_HASH, len(payload), avg_dopamine, 0) + payload
        sock.sendto(packet, (GENESIS_IP, PORT_OUT))

        # Читаем Моторные выходы
        try:
            data, _ = sock.recvfrom(65535)
            if len(data) >= 20:
                magic, z_hash, m_hash, p_size, reward, _ = struct.unpack("<IIIIhH", data[:20])

                if magic == GSOO_MAGIC and z_hash == MOTOR_ZONE_HASH:
                    payload = data[20:20+p_size]
                    
                    # [DOD] Декодируем весь батч целиком (Time Integration)
                    # 100 тиков * 256 нейронов (1 байт на нейрон)
                    spikes = np.frombuffer(payload, dtype=np.uint8).reshape((BATCH_TICKS, 256))
                    total_spikes = np.sum(spikes, axis=0, dtype=np.int32) # Сумма спайков каждого нейрона за 10 мс
                    
                    for i in range(8):
                        flexor = np.sum(total_spikes[(i*2)*16 : (i*2+1)*16])
                        extensor = np.sum(total_spikes[(i*2+1)*16 : (i*2+2)*16])
                        # Population Rate Code (нормализуем и масштабируем)
                        state.action[i] = ((flexor - extensor) / (BATCH_TICKS * 16.0)) * 2.0

                    if random.random() < 0.1:
                        print(f"💉 [Client] Reward: {reward}, Spikes: {np.sum(total_spikes)}, Action: {state.action[0]:.3f}")
        except socket.timeout:
            pass

def main():
    try:
        env = gym.make('Ant-v4', render_mode="human", exclude_current_positions_from_observation=False)
        print("📺 [Client] Rendering in HUMAN mode")
    except Exception as e:
        env = gym.make('Ant-v4', render_mode="rgb_array", exclude_current_positions_from_observation=False)

    state.obs, _ = env.reset()
    state.last_x = state.obs[0]

    udp_thread = threading.Thread(target=udp_hot_loop)
    udp_thread.start()

    try:
        while True:
            state.obs, reward, terminated, truncated, info = env.step(state.action)

            current_x = state.obs[0]
            if current_x > state.last_x + 0.005:
                state.steps_no_progress = 0
            else:
                state.steps_no_progress += 1

            flipped = state.obs[2] < 0.25
            stuck = state.steps_no_progress > 200

            if terminated or truncated or flipped or stuck:
                state.obs, _ = env.reset()
                state.steps_no_progress = 0
                state.last_x = state.obs[0]

    except KeyboardInterrupt:
        state.running = False
        udp_thread.join()
        env.close()

if __name__ == "__main__":
    main()

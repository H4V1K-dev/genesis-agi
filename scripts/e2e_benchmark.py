#!/usr/bin/env python3
import socket
import struct
import time
import numpy as np
import threading

# FNV-1a 32-bit (Строго совпадает с genesis-core::hash)
def fnv1a_32(data: bytes) -> int:
    hash_value = 0x811c9dc5
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value

# Константы протокола (Spec 08 §2.7)
GSIO_MAGIC = 0x4F495347 # "GSIO"
GSOO_MAGIC = 0x4F4F5347 # "GSOO"
HEADER_FMT = "<IIIIhH"  # magic, zone, matrix, size, reward, padding (20 bytes)

# Настройки PingPongBrain
ZONE_IN = fnv1a_32(b"SensoryCortex")
MAT_IN  = fnv1a_32(b"sensory_in")
ZONE_OUT = fnv1a_32(b"MotorCortex")
MAT_OUT = fnv1a_32(b"motor_out")

TARGET_IP = "127.0.0.1"
PORT_IN = 8081  # Нода слушает входы здесь
PORT_OUT = 8082 # Мы слушаем выходы здесь

# 16x16 = 256 бит = 8 u32 слов на тик. バтч = 100 тиков.
BATCH_TICKS = 100
INPUT_PAYLOAD_SIZE = 8 * 4 * BATCH_TICKS  # 3200 bytes
OUTPUT_PAYLOAD_SIZE = 256 * BATCH_TICKS   # 25600 bytes

def recv_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT_OUT))
    sock.settimeout(10.0) # Increased for initialization
    
    print(f"[RX] Listening for GSOO packets on {PORT_OUT}...")
    batches_received = 0
    start_time = time.time()
    
    try:
        while batches_received < 100: # Ждем 100 батчей (10 секунд симуляции)
            data, _ = sock.recvfrom(65535)
            if len(data) < 20: continue
            
            # Распаковка заголовка ExternalIoHeader
            magic, z_hash, m_hash, p_size, reward, _ = struct.unpack(HEADER_FMT, data[:20])
            
            if magic != GSOO_MAGIC: continue
            if p_size != OUTPUT_PAYLOAD_SIZE:
                print(f"[!] Warning: Expected {OUTPUT_PAYLOAD_SIZE} bytes, got {p_size}")
                continue
                
            # Валидация данных: проверяем что-то кроме нулей
            payload = np.frombuffer(data[20:20+p_size], dtype=np.uint8)
            spikes = np.sum(payload)
            
            batches_received += 1
            if batches_received % 10 == 0:
                print(f"[RX] Batch {batches_received}/100 | Somas fired: {spikes}")
                
    except socket.timeout:
        print("[!] RX Timeout. Pipeline broke or node died.")
        
    elapsed = time.time() - start_time
    print(f"[RX] Completed. {batches_received} batches in {elapsed:.3f}s (Target: ~1.000s)")

def send_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Запускаем "квадрат" по кругу
    pulse_idx = 0
    
    for _ in range(100):
        # Строим плоскую маску в памяти (Zero-Allocation on Python side via NumPy)
        # 100 тиков * 8 u32
        bitmask = np.zeros((BATCH_TICKS, 8), dtype=np.uint32)
        
        # Каждые 10 тиков бьем в пиксель (pulse_idx)
        word_idx = (pulse_idx // 32)
        bit_idx = pulse_idx % 32
        bitmask[0:10, word_idx] |= (1 << bit_idx) 
        
        pulse_idx = (pulse_idx + 7) % 256
        
        # Формируем пакет 
        payload = bitmask.tobytes()
        header = struct.pack(HEADER_FMT, GSIO_MAGIC, ZONE_IN, MAT_IN, len(payload), 0, 0)
        packet = header + payload
        
        sock.sendto(packet, (TARGET_IP, PORT_IN))
        time.sleep(0.01) # Отправляем 100 батчей/сек (эмуляция реального времени)

if __name__ == "__main__":
    rx = threading.Thread(target=recv_loop)
    rx.start()
    
    time.sleep(0.5) # Даем RX запуститься
    print("[TX] Starting injection...")
    send_loop()
    
    rx.join()
    print("[E2E] Benchmark finished.")

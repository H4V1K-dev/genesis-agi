import pygame
import socket
import struct
import numpy as np
import threading
import time

def fnv1a_32(data: bytes) -> int:
    hash_value = 0x811c9dc5
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value

ZONE_HASH_IN = fnv1a_32(b"SensoryCortex")
MATRIX_HASH_IN = fnv1a_32(b"sensory_in")

ZONE_HASH_OUT = fnv1a_32(b"MotorCortex")
MATRIX_HASH_OUT = fnv1a_32(b"motor_out")

GENESIS_IP = "127.0.0.1"
PORT_OUT = 8081 # Send point (SensoryCortex)
PORT_IN = 8092  # Receive point (MotorCortex)

class State:
    def __init__(self):
        self.running = True
        self.sensory = np.zeros((16, 16), dtype=np.uint8)
        self.motor_brightness = np.zeros((16, 16), dtype=np.uint8)
        
        self.mode = "NORMAL" # NORMAL, RECORDING, PLAYING
        self.macro_frames = []
        self.play_idx = 0

state = State()
# Автозаливка: полный белый квадрат (1 на всех пикселях) для тестирования без мыши.
# Нейроны на Node A получат синаптический ввод сразу при старте.
state.sensory.fill(1)
print("[AUTO] Sensory matrix pre-filled with ALL-ONES for pipeline test")


def udp_hot_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # СТРОГО BIND ДО ЛЮБОЙ ОТПРАВКИ — фиксируем наш порт как 8092
    sock.bind(("127.0.0.1", PORT_IN))
    # Non-blocking mode — drain loop будет вычитывать буфер досуха
    sock.setblocking(False)
    
    # Максимизируем буфер приёма ОС (4MB, для 64MB/s потока)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except Exception:
        pass

    NUM_CHANNELS = 256
    BATCH_TICKS = 100
    EXPECTED_PAYLOAD = NUM_CHANNELS * BATCH_TICKS  # 25600
    EXPECTED_PACKET = 12 + EXPECTED_PAYLOAD          # 25612 (header + payload)
    
    frame_count = 0
    drop_count = 0
    
    while state.running:
        # 1. SEND INPUT (bitmask для Node A)
        current_frame = state.sensory
        if state.mode == "PLAYING" and len(state.macro_frames) > 0:
            current_frame = state.macro_frames[state.play_idx]
            state.play_idx = (state.play_idx + 1) % len(state.macro_frames)
        elif state.mode == "RECORDING":
            state.macro_frames.append(current_frame.copy())
            
        flat = current_frame.flatten()
        words = []
        for i in range(8):
            word = 0
            for bit in range(32):
                if flat[i*32 + bit]:
                    word |= (1 << bit)
            words.append(word)
            
        payload_words = words * 100
        payload = struct.pack(f"<IIIhH{len(payload_words)}I", ZONE_HASH_IN, MATRIX_HASH_IN, len(payload_words)*4, 0, 0, *payload_words)
        try:
            sock.sendto(payload, (GENESIS_IP, PORT_OUT))
        except BlockingIOError:
            pass
        
        # 2. DRAIN LOOP: Вычитываем буфер ОС ДОСУХА, оставляем только последний кадр.
        # При 2500 пакетов/сек и PyGame 30 FPS, за 33мс накапливается ~83 пакета.
        # Мы выбрасываем все промежуточные, рендерим только самый свежий.
        latest_data = None
        drained = 0
        while True:
            try:
                data, addr = sock.recvfrom(65535)
                drained += 1
                if len(data) >= EXPECTED_PACKET:
                    latest_data = data
                elif len(data) > 1:
                    drop_count += 1
            except BlockingIOError:
                break  # Буфер пуст — выходим
            except Exception:
                break
        
        if latest_data is not None:
            frame_count += 1
            header = struct.unpack("<III", latest_data[:12])
            if header[0] == ZONE_HASH_OUT and header[1] == MATRIX_HASH_OUT:
                payload_bytes = latest_data[12:12 + EXPECTED_PAYLOAD]
                spike_matrix = np.frombuffer(payload_bytes, dtype=np.uint8).reshape(BATCH_TICKS, NUM_CHANNELS)
                sum_spikes = spike_matrix.sum(axis=0).astype(np.int32)
                brightness = np.clip(sum_spikes * 2.55, 0, 255).astype(np.uint8)
                state.motor_brightness = brightness.reshape((16, 16))
                if frame_count % 30 == 1:
                    print(f"[RENDER] frame={frame_count} drained={drained} max_brightness={brightness.max()} active={np.count_nonzero(sum_spikes)}")
            
        time.sleep(0.02)  # ~50 Hz send rate

def main():
    pygame.init()
    WIDTH, HEIGHT = 640, 360
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Genesis PingPong Harness")
    font = pygame.font.SysFont(None, 24)
    
    udp_thread = threading.Thread(target=udp_hot_loop)
    udp_thread.start()
    
    drawing = False
    erasing = False
    clock = pygame.time.Clock()
    
    while state.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    state.mode = "RECORDING"
                    state.macro_frames = []
                elif event.key == pygame.K_s: 
                    state.mode = "NORMAL"
                elif event.key == pygame.K_p: 
                    if state.macro_frames:
                        state.mode = "PLAYING"
                        state.play_idx = 0
                elif event.key == pygame.K_c: 
                    state.sensory.fill(0)
                    state.mode = "NORMAL"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if 330 <= mx < 330 + 16*18 and 40 <= my < 40 + 16*18:
                    drawing = event.button == 1
                    erasing = event.button == 3
                    x = (mx - 330) // 18
                    y = (my - 40) // 18
                    state.sensory[y, x] = 1 if drawing else 0
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                erasing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing or erasing:
                    mx, my = event.pos
                    if 330 <= mx < 330 + 16*18 and 40 <= my < 40 + 16*18:
                        x = (mx - 330) // 18
                        y = (my - 40) // 18
                        state.sensory[y, x] = 1 if drawing else 0

        screen.fill((30, 30, 30))
        cell_size = 18
        
        # Draw Motor
        for y in range(16):
            for x in range(16):
                c = state.motor_brightness[y, x]
                color = (c, c, c)
                pygame.draw.rect(screen, color, (20 + x*cell_size, 40 + y*cell_size, cell_size-1, cell_size-1))
                
        # Draw Sensory
        for y in range(16):
            for x in range(16):
                c = 255 if state.sensory[y, x] else 0
                color = (c, 0, 0) if c else (40, 40, 40)
                pygame.draw.rect(screen, color, (330 + x*cell_size, 40 + y*cell_size, cell_size-1, cell_size-1))
                
        text_motor = font.render("Motor Output (L5_6)", True, (200, 200, 200))
        screen.blit(text_motor, (20, 340))
        text_sensory = font.render("Sensory Input (L4)", True, (200, 200, 200))
        screen.blit(text_sensory, (330, 340))
        
        mode_color = (0, 255, 0)
        if state.mode == "RECORDING": mode_color = (255, 0, 0)
        elif state.mode == "PLAYING": mode_color = (0, 200, 255)
        
        text_mode = font.render(f"MODE: {state.mode} (R: Record, S: Stop, P: Play, C: Clear)", True, mode_color)
        screen.blit(text_mode, (20, 10))
        
        pygame.display.flip()
        clock.tick(30)
        
    udp_thread.join()
    pygame.quit()

if __name__ == "__main__":
    main()

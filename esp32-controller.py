import threading
import cv2
import numpy as np
import time
import mediapipe as mp
import socket
from PIL import Image

esp32 = { 
    "esp32_1": {
        "ip": "172.20.10.2",
        "port": 1234,
        "camera_index": 1,
        "logo_path": "other-side_logo-fill.png"
    },
    "esp32_2": {
        "ip": "172.20.10.3",
        "port": 1235,
        "camera_index": 0,
        "logo_path": "other-side_logo-fill.png"
    }
}


PANEL_WIDTH, PANEL_HEIGHT = 16, 16
TOTAL_WIDTH, TOTAL_HEIGHT = PANEL_WIDTH * 2, PANEL_HEIGHT * 2
NUM_PIXELS = TOTAL_WIDTH * TOTAL_HEIGHT
BRIGHTNESS_MIN = 10
SEND_INTERVAL = 1 / 1

mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

def connect_to_esp32(esp32_ip, esp32_port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect((esp32_ip, esp32_port))
        return sock
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def zoom_frame(frame, zoom_factor=1.5):
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def load_logo(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((TOTAL_WIDTH, TOTAL_HEIGHT))
    new_img = Image.new("RGB", (TOTAL_WIDTH, TOTAL_HEIGHT), (0, 0, 0))
    offset_x = (TOTAL_WIDTH - img.size[0]) // 2
    offset_y = (TOTAL_HEIGHT - img.size[1]) // 2
    new_img.paste(img, (offset_x, offset_y))
    return np.array(new_img)

def rearrange(image):
    PANEL_OFFSETS = [(16, 16), (0, 16), (0, 0), (16, 0)]
    output = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
    for panel_idx, (ox, oy) in enumerate(PANEL_OFFSETS):
        panel = image[oy:oy + PANEL_HEIGHT, ox:ox + PANEL_WIDTH]
        if panel_idx in [0, 1]:
            panel = np.rot90(panel, k=0.5, axes=(0, 1))
        elif panel_idx in [2, 3]:
            panel = np.flip(panel, axis=1)
            panel = np.flip(panel, axis=0)
            panel = np.rot90(panel, k=0.5, axes=(0, 1))
        output[oy:oy + PANEL_HEIGHT, ox:ox + PANEL_WIDTH] = panel
    return np.flip(output, axis=1)


def main(sock, camera_index, logo_path, name):
        
    try:
        cap = cv2.VideoCapture(camera_index)
        logo_np = load_logo(logo_path)
        last_send_time = 0
        presence_detected = False
        show_silhouette = False
        last_state_change_time = time.time()
    except Exception as e:
        print(f"[{name}] Send error: {e}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame = zoom_frame(frame, zoom_factor=2.5)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(frame_rgb)
        if results.segmentation_mask is not None:
            mask = (results.segmentation_mask > 0.5).astype(np.uint8)
        else:
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        presence_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
        current_time = time.time()

        if presence_ratio > 0.05:
            if not presence_detected:
                presence_detected = True
                last_state_change_time = current_time
            if not show_silhouette and (current_time - last_state_change_time) >= 1:
                show_silhouette = True
        else:
            if presence_detected:
                presence_detected = False
                last_state_change_time = current_time
            if show_silhouette and (current_time - last_state_change_time) >= 2:
                show_silhouette = False

        silhouette = cv2.bitwise_and(frame, frame, mask=mask)
        resized = cv2.resize(silhouette, (TOTAL_WIDTH, TOTAL_HEIGHT))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        gamma = 1.5
        enhanced = np.power(enhanced / 255, gamma) * 255
        enhanced = np.clip(enhanced, BRIGHTNESS_MIN, 255).astype(np.uint8)
        normalized = enhanced

        fg_color, bg_color = np.array([229, 0, 68]), np.array([33, 45, 148])
        mask_resized = cv2.resize(mask, (TOTAL_WIDTH, TOTAL_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(bool)
        colored = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
        colored[mask_resized] = (normalized[mask_resized, None] * (fg_color / 255)).astype(np.uint8)
        colored[~mask_resized] = bg_color

        final_img = colored if show_silhouette else logo_np
        rearranged = rearrange(final_img)
        flat_data = rearranged.flatten().tolist()

        if time.time() - last_send_time >= SEND_INTERVAL:
            try:
                print(f"[{name}] Sending data...")
                sock.send(bytearray(flat_data))
                last_send_time = time.time()
            except Exception as e:
                print(f"[{name}] Send error: {e}")

def launch_main_for_esp32(esp32_name, esp32_info):
    print(f"[{esp32_name}] Connecting to ESP32...")
    sock = connect_to_esp32(esp32_info["ip"], esp32_info["port"])
    if sock:
        print(f"[{esp32_name}] Connected.")
        main(sock, esp32_info["camera_index"], esp32_info["logo_path"], esp32_name)
    else:
        print(f"[{esp32_name}] Connection failed.")

threads = []

for esp32_name, esp32_info in esp32.items():
    t = threading.Thread(target=launch_main_for_esp32, args=(esp32_name, esp32_info))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

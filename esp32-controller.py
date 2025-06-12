import threading
import cv2
import numpy as np
import time
import mediapipe as mp
import socket
from PIL import Image
import queue

# --- Config ESP32 ---
ESP32_CONFIGS = {
    "esp32_1": {
        "ip": "172.20.10.3", 
        "port": 1234, 
        "camera_index": 1, 
        "logo_path": 
        "other-side_logo-fill.png"},
    "esp32_2": {
        "ip": "172.20.10.2", 
        "port": 1235, 
        "camera_index": 0, 
        "logo_path": "other-side_logo-fill.png"
    }
}

# --- Constantes ---
PANEL_WIDTH, PANEL_HEIGHT      = 16, 16
TOTAL_WIDTH, TOTAL_HEIGHT      = PANEL_WIDTH*2, PANEL_HEIGHT*2
NUM_PIXELS                     = TOTAL_WIDTH * TOTAL_HEIGHT
BRIGHT_MIN                     = 10
FPS                            = 15
SEND_INTERVAL                  = 1 / FPS

mp_selfie = mp.solutions.selfie_segmentation

# Queue pour transmettre (name, preview_frame, status) au GUI thread
preview_queue = queue.Queue()
# Event pour arrêter tous les threads proprement
stop_event = threading.Event()

def connect_to_esp32(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect((ip, port))
        return sock
    except Exception as e:
        print(f"→ Connexion échouée à {ip}:{port} ({e})")
        return None

def zoom_frame(frame, zoom_factor=1.5):
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]

    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

def load_logo(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((TOTAL_WIDTH, TOTAL_HEIGHT))
    bg = Image.new("RGB", (TOTAL_WIDTH, TOTAL_HEIGHT), (0,0,0))
    bx, by = (TOTAL_WIDTH - img.width)//2, (TOTAL_HEIGHT - img.height) // 2
    bg.paste(img, (bx, by))
    return np.array(bg)

def rearrange(image):
    PANEL_OFFSETS = [(16, 16), (0, 16), (0, 0), (16, 0)]
    output = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
    for panel_idx, (ox, oy) in enumerate(PANEL_OFFSETS):
        panel = image[oy:oy + PANEL_HEIGHT, ox:ox + PANEL_WIDTH]
        if panel_idx in [0, 1]:
            panel = np.rot90(panel, k=0.5, axes=(0, 1))  # k0.5 = valeur interdite
            # panel = np.rot90(panel, k=1, axes=(0, 1))  # k0.5 = valeur interdite


        elif panel_idx in [2, 3]:
            panel = np.flip(panel, axis=1)
            panel = np.flip(panel, axis=0)
            panel = np.rot90(panel, k=0.5, axes=(0, 1))
            # panel = np.rot90(panel, k=2, axes=(0, 1))  
        output[oy:oy + PANEL_HEIGHT, ox:ox + PANEL_WIDTH] = panel
    return np.flip(output, axis=1)

def esp32_worker(name, cfg):
    """Thread qui capture, segmente, encode et envoie,
       et pousse un preview dans la queue."""
    ip, port = cfg["ip"], cfg["port"]
    cam_idx, logo_path = cfg["camera_index"], cfg["logo_path"]

    sock = connect_to_esp32(ip, port)
    status = "Connected" if sock else "Disconnected"
    logo = load_logo(logo_path)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[{name}] Caméra {cam_idx} indisponible.")
        return

    last_send = time.time()
    presence = False
    show_silh = False
    last_sw = time.time()

    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            # — segmentation & transformation —
            frame = cv2.flip(frame, 1)
            frame = zoom_frame(frame, zoom_factor=2.5)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = segmenter.process(rgb)
            mask = (res.segmentation_mask>0.5).astype(np.uint8) if res.segmentation_mask is not None else np.zeros(frame.shape[:2], np.uint8)

            ratio = mask.sum()/(mask.shape[0]*mask.shape[1])
            now = time.time()
            if ratio > 0.05:
                if not presence:
                    presence, last_sw = True, now
                if not show_silh and now - last_sw >= 1:
                    show_silh = True
            else:
                if presence:
                    presence, last_sw = False, now
                if show_silh and now - last_sw >= 2:
                    show_silh = False

            sil = cv2.bitwise_and(frame, frame, mask=mask)
            small = cv2.resize(sil, (TOTAL_WIDTH, TOTAL_HEIGHT))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4,4))
            enh = clahe.apply(gray)
            enh = np.clip((enh/255)**1.5 * 255, BRIGHT_MIN, 255).astype(np.uint8)

            fg, bg = np.array([229,0,68]), np.array([33,45,148])
            m2 = cv2.resize(mask, (TOTAL_WIDTH,TOTAL_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(bool)
            colored = np.zeros((TOTAL_HEIGHT,TOTAL_WIDTH,3), np.uint8)
            colored[m2] = (enh[m2,None] * (fg/255)).astype(np.uint8)
            colored[~m2] = bg

            final = colored if show_silh else logo
            buf = rearrange(final)
            data = (buf >>2).astype(np.uint8)
            flat_data = data.flatten().tolist()

            # — envoi ESP32 —
            if now - last_send >= SEND_INTERVAL and sock:
                try:
                    sock.send(bytearray(flat_data))
                    status = "Connected"
                    last_send = now
                except Exception:
                    sock.close()
                    sock = connect_to_esp32(ip, port)
                    status = "Reconnecting..." if sock else "Disconnected"

            # — push preview dans la queue —
            disp = enh if show_silh else gray
            preview = cv2.resize(disp, (TOTAL_WIDTH*10, TOTAL_HEIGHT*10), interpolation=cv2.INTER_NEAREST)
            preview_queue.put((name, preview, status))

    # cleanup de fin de thread
    if sock:
        try:
            sock.send(bytearray([0]*NUM_PIXELS*3))
        except: pass
        sock.close()
    cap.release()
    print(f"[{name}] Thread terminé.")

def gui_loop():
    """Thread GUI unique, lit preview_queue, affiche et gère 'q'."""
    windows = set()
    previews = {}  # name -> (frame, status)
    while not stop_event.is_set():
        try:
            # récupère toutes les nouvelles previews
            while True:
                name, frame, status = preview_queue.get_nowait()
                previews[name] = (frame, status)
        except queue.Empty:
            pass

        # affiche chaque fenêtre
        for name, (frame, status) in previews.items():
            win = f"Preview {name}"
            if win not in windows:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                windows.add(win)
            disp = frame.copy()
            cv2.putText(disp, f"{name}: {status}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow(win, disp)

        # touche 'q' pour tout arrêter
        if cv2.waitKey(50) & 0xFF == ord('q'):
            stop_event.set()
            break

    # fermer toutes les fenêtres
    for win in list(windows):
        cv2.destroyWindow(win)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 1) lancer les workers ESP32 en daemon threads
    workers = []
    for name, cfg in ESP32_CONFIGS.items():
        t = threading.Thread(target=esp32_worker, args=(name, cfg), daemon=True)
        t.start()
        workers.append(t)

    # 2) exécuter le GUI LOOP dans le thread principal (obligatoire pour OpenCV)
    try:
        gui_loop()
    except Exception as e:
        print(f"Erreur GUI : {e}")
    finally:
        # signale aux workers qu'il faut s'arrêter
        stop_event.set()
        # attend la fin des workers
        for t in workers:
            t.join()
        print("Terminé.")
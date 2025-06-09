import cv2
import numpy as np
import time
import mediapipe as mp
import socket
from PIL import Image

### --- CONFIGURATION ---
ESP32_IP = '172.20.10.2'  # IP de l'ESP32
ESP32_PORT = 1234        # TCP port
PANEL_WIDTH = 16
PANEL_HEIGHT = 16
NUM_PANELS = 4
TOTAL_WIDTH = PANEL_WIDTH * 2
TOTAL_HEIGHT = PANEL_HEIGHT * 2
NUM_PIXELS = TOTAL_WIDTH * TOTAL_HEIGHT
BRIGHTNESS_MIN = 10
SEND_INTERVAL = 1 / 15    # ≈15 FPS
DELAY_IN_SECONDS = 3     # délai avant d'afficher la silhouette
DELAY_OUT_SECONDS = 3    # délai avant de revenir au logo
presence_detected = False
last_state_change_time = time.time()
show_silhouette = False


# Offsets des panneaux (de 0 à 16 sur une matrice 16x16)
PANEL_OFFSETS = [
    (16, 16), 
    (0, 16), 
    (0, 0),
    (16, 0)
]

# Initialisation MediaPipe
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

# Webcam
cap = cv2.VideoCapture(1)
# Zoom 
# cap.set(cv2.CAP_PROP_ZOOM, 50)
last_send_time = 0
connection_status = "Disconnected"

# Initialisation socket TCP
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)  # Timeout de 1 seconde

def load_logo_with_padding(path, target_size=(TOTAL_WIDTH, TOTAL_HEIGHT), bg_color=(0, 0, 0)):
    # Charger et convertir l'image
    img = Image.open(path).convert("RGB")
    
    # Calculer le ratio de redimensionnement
    img.thumbnail(target_size)
    new_img = Image.new("RGB", target_size, bg_color)
    
    # Centrage de l'image
    offset_x = (target_size[0] - img.size[0]) // 2
    offset_y = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (offset_x, offset_y))
    
    return np.array(new_img)

def zoom_frame(frame, zoom_factor=1.5):
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    
    cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed

def connect_to_esp32():
    try:
        sock.connect((ESP32_IP, ESP32_PORT))
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def rearrange_for_panels(image):
    output = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)

    for panel_idx, (offset_x, offset_y) in enumerate(PANEL_OFFSETS):
        panel = image[offset_y:offset_y + PANEL_HEIGHT, offset_x:offset_x + PANEL_WIDTH]

        # Appliquer la bonne orientation par panneau
        if panel_idx in [0, 1]:  # Bottom right & bottom left
            panel = np.rot90(panel, k=0.5, axes=(0, 1))  # Rotate 90 degrees
        elif panel_idx in [2, 3]:  # Top left & top right
            panel = np.flip(panel, axis=1)  # Horizontal flip
            panel = np.flip(panel, axis=0)  # Horizontal flip
            panel = np.rot90(panel, k=0.5, axes=(0, 1))  # Rotate 90 degrees

        output[offset_y:offset_y + PANEL_HEIGHT, offset_x:offset_x + PANEL_WIDTH] = panel

    return output

# Connexion à l'ESP32
if not connect_to_esp32():
    print("Failed to connect to ESP32. Please check the IP address and make sure ESP32 is running.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = zoom_frame(frame, zoom_factor=2.5) 

    # Segmentation de la silhouette
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
    results = segmenter.process(frame_rgb)
    mask = (results.segmentation_mask > 0.5).astype(np.uint8)

    # Calcul du pourcentage de pixels détectés comme silhouette
    presence_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])

    # Seuil de détection 
    PRESENCE_THRESHOLD = 0.05  # 5 % de la zone

    # Détection présence
    current_time = time.time()

    if presence_ratio > PRESENCE_THRESHOLD:
        if not presence_detected:
            presence_detected = True
            last_state_change_time = current_time
        # attendre DELAY_IN_SECONDS avant d'afficher la silhouette
        if not show_silhouette and (current_time - last_state_change_time) >= DELAY_IN_SECONDS:
            show_silhouette = True
    else:
        if presence_detected:
            presence_detected = False
            last_state_change_time = current_time
        # attendre DELAY_OUT_SECONDS avant de revenir au logo
        if show_silhouette and (current_time - last_state_change_time) >= DELAY_OUT_SECONDS:
            show_silhouette = False


    # Appliquer masque
    silhouette = cv2.bitwise_and(frame, frame, mask=mask)

    # Redimensionner + convertir en niveaux de gris
    resized = cv2.resize(silhouette, (TOTAL_WIDTH, TOTAL_HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Appliquer un minimum de luminosité
    #normalized = np.interp(gray, [0, 255], [BRIGHTNESS_MIN, 155]).astype(np.uint8)

    # Appliquer CLAHE (contraste local) pour plus de profondeur
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Appliquer un minimum de luminosité
    gamma = 1.5
    enhanced = np.power(enhanced / 255, gamma) * 255
    enhanced = np.clip(enhanced, BRIGHTNESS_MIN, 255).astype(np.uint8)

    # Normaliser
    normalized = enhanced

    # Charger et redimensionner le logo
    logo_np = load_logo_with_padding("other-side_logo-fill.png")

    # Appliquer une couleur sur la silhouette et une autre sur le fond
    USE_COLOR = True  # False = niveaux de gris, True = rose
    if USE_COLOR:
        # fg_color = np.array([229, 0, 68])   # silhouette : #E50044 (RGB)
        # bg_color = np.array([33, 45, 148])  # fond : #4156A2 (RGB)

        fg_color = np.array([33, 45, 148])   # silhouette : #E50044 (RGB)
        bg_color = np.array([229, 0, 68])  # fond : #4156A2 (RGB)

        # Redimensionner le masque à la taille finale
        mask_resized = cv2.resize(mask, (TOTAL_WIDTH, TOTAL_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_resized.astype(bool)

        # Appliquer couleur en fonction du masque
        silhouette_rgb = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
        silhouette_rgb[mask_resized] = (normalized[mask_resized, None] * (fg_color / 255)).astype(np.uint8)
        silhouette_rgb[~mask_resized] = bg_color
    else:
        silhouette_rgb = np.stack([normalized]*3, axis=-1).astype(np.uint8)


    if show_silhouette:
        grayscale_rgb = silhouette_rgb
    else:
        grayscale_rgb = logo_np

    # Réorganiser selon la disposition des panneaux
    rearranged = rearrange_for_panels(grayscale_rgb)
    rearranged = np.flip(rearranged, axis=1)
    
    # Aplatir pour envoi
    flat_data = rearranged.flatten().tolist()

    # Envoi TCP vers ESP32
    if time.time() - last_send_time >= SEND_INTERVAL:
        try:
            # Envoyer le paquet
            sock.send(bytearray(flat_data))
            last_send_time = time.time()
            connection_status = "Connected"
            print(f"[→] Sent {len(flat_data)} RGB bytes")
        except Exception as e:
            connection_status = f"Error: {str(e)}"
            print(f"[✗] Envoi échoué : {e}")
            # Essayer de reconnecter
            try:
                sock.close()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                if connect_to_esp32():
                    print("Reconnected to ESP32")
            except Exception as reconnect_error:
                print(f"Reconnection failed: {reconnect_error}")

    # Affichage debug
    preview = cv2.resize(normalized, (TOTAL_WIDTH * 10, TOTAL_HEIGHT * 10), interpolation=cv2.INTER_NEAREST)
    cv2.putText(preview, connection_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.flip(preview, 1, preview)
    cv2.imshow("Silhouette niveau de gris", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
black_frame = bytearray([0] * NUM_PIXELS * 3)
try:
    sock.send(black_frame)
    print("LEDs éteintes.")
except:
    print("Impossible d’envoyer l’image noire.")
sock.close()
cap.release()
cv2.destroyAllWindows()
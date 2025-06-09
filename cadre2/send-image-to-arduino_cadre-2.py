import cv2
import numpy as np
import time
import mediapipe as mp
import socket

# CONFIGURATION
ESP32_IP = '172.20.10.3'  # IP de l'ESP32
ESP32_PORT = 1235       # TCP port
PANEL_WIDTH = 16
PANEL_HEIGHT = 16
NUM_PANELS = 4
TOTAL_WIDTH = PANEL_WIDTH * 2
TOTAL_HEIGHT = PANEL_HEIGHT * 2
NUM_PIXELS = TOTAL_WIDTH * TOTAL_HEIGHT
BRIGHTNESS_MIN = 10
SEND_INTERVAL = 1 / 15   # ≈15 FPS

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
cap = cv2.VideoCapture(2)
last_send_time = 0
connection_status = "Disconnected"

# Initialisation socket TCP
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)  # Timeout de 1 seconde

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

    # Segmentation de la silhouette
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
    results = segmenter.process(frame_rgb)
    mask = (results.segmentation_mask > 0.5).astype(np.uint8)

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

    # Appliquer une couleur sur la silhouette et une autre sur le fond
    USE_COLOR = True  # False = niveaux de gris, True = rose
    if USE_COLOR:
        fg_color = np.array([229, 0, 68])   # silhouette : #E50044 (RGB)
        bg_color = np.array([33, 45, 148])  # fond : #4156A2 (RGB)

        # fg_color = np.array([33, 45, 148])   # silhouette : #E50044 (RGB)
        # bg_color = np.array([229, 0, 68])  # fond : #4156A2 (RGB)

        # Redimensionner le masque à la taille finale
        mask_resized = cv2.resize(mask, (TOTAL_WIDTH, TOTAL_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_resized.astype(bool)

        # Appliquer couleur en fonction du masque
        silhouette_rgb = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
        silhouette_rgb[mask_resized] = (normalized[mask_resized, None] * (fg_color / 255)).astype(np.uint8)
        silhouette_rgb[~mask_resized] = bg_color
    else:
        silhouette_rgb = np.stack([normalized]*3, axis=-1).astype(np.uint8)


    # Réorganiser selon la disposition des panneaux
    rearranged = rearrange_for_panels(silhouette_rgb)
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
    cv2.flip(preview, 1, preview)
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
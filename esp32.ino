#include <WiFi.h>
#include <FastLED.h>

// WiFi identifiants
const char* ssid = "iPhone de Christelle";
const char* password = "chuu1002";

// LED configuration
#define PANEL_WIDTH 16
#define PANEL_HEIGHT 16
#define NUM_PANELS 4
#define NUM_LEDS (PANEL_WIDTH * PANEL_HEIGHT * NUM_PANELS)  // 1024 LEDs
#define LED_PIN 16  // Pin données sur l'ESP32 
#define LED_TYPE WS2812B
#define COLOR_ORDER GRB

// TCP configuration
#define TCP_PORT 1235
WiFiServer server(TCP_PORT);
WiFiClient client;

// LED tableau
CRGB leds[NUM_LEDS];

// Panel offsets (en LEDs)
const int PANEL_OFFSETS[NUM_PANELS][2] = {
   {16, 16}, // Panel 0 (bottom right)
   {0, 16},  // Panel 1 (bottom left)
   {0, 0},   // Panel 2 (top left)
   {16, 0}   // Panel 3 (top right)
};

// Tableau de correspondance coordonnées -> index LED (LUT)
int xyToLedIndex[32][32];

void precalculateXYToIndex() {
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            int panel = -1;
            for (int p = 0; p < NUM_PANELS; p++) {
                if (x >= PANEL_OFFSETS[p][0] && x < PANEL_OFFSETS[p][0] + PANEL_WIDTH &&
                    y >= PANEL_OFFSETS[p][1] && y < PANEL_OFFSETS[p][1] + PANEL_HEIGHT) {
                    panel = p;
                    break;
                }
            }
            
            if (panel == -1) {
                xyToLedIndex[x][y] = -1;
                continue;
            }
            
            int localX = x - PANEL_OFFSETS[panel][0];
            int localY = y - PANEL_OFFSETS[panel][1];
            
            int index;
            if (localY % 2 == 0) {
                index = localY * PANEL_WIDTH + localX;
            } else {
                index = localY * PANEL_WIDTH + (PANEL_WIDTH - 1 - localX);
            }
            
            xyToLedIndex[x][y] = panel * (PANEL_WIDTH * PANEL_HEIGHT) + index;
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000); 
    Serial.println("Setup started ✔️");
    
    precalculateXYToIndex();
    Serial.println("LED mapping table initialized");
    
    FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
    FastLED.setBrightness(60);
    FastLED.setDither(false);  // Désactive le dithering car on envoie des images complètes
    FastLED.clear();
    FastLED.show();
    Serial.println("LEDs initialized");

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    server.begin();
    Serial.print("TCP Server started on port ");
    Serial.println(TCP_PORT);
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi connection lost. Reconnecting...");
        WiFi.reconnect();
        delay(5000);
        return;
    }

    if (!client.connected()) {
        client = server.available();
        if (client) {
            Serial.println("New client connected");
        }
    } else {
        static uint8_t buffer[NUM_LEDS * 3];

        // Si on a assez de données pour une image complète, on la lit
        if (client.available() >= NUM_LEDS * 3) {
            size_t bytesRead = client.read(buffer, NUM_LEDS * 3);
            
            // On traite l'image seulement si on a reçu exactement ce qu'il faut
            if (bytesRead == NUM_LEDS * 3) {
                for (int y = 0; y < PANEL_HEIGHT * 2; y++) {
                    for (int x = 0; x < PANEL_WIDTH * 2; x++) {
                        int ledIndex = xyToLedIndex[x][y];  // Utilisation directe de la LUT
                        if (ledIndex >= 0 && ledIndex < NUM_LEDS) {  // On garde juste cette vérification
                            int bufferIndex = (y * PANEL_WIDTH * 2 + x) * 3;
                            leds[ledIndex] = CRGB(
                                buffer[bufferIndex],
                                buffer[bufferIndex + 1],
                                buffer[bufferIndex + 2]
                            );
                        }
                    }
                }
                FastLED.show();
            }
            // Si on a reçu moins, on ignore simplement et on attendra la prochaine image
            // TCP garantit que les données suivantes seront dans le bon ordre
        }

        if (!client.connected()) {
            Serial.println("Client disconnected");
            client.stop();
        }
    }

    delay(10); 
}
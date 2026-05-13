import cv2
import mediapipe as mp
import numpy as np
import math
import os
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURACIÓN DE AUDIO (A PRUEBA DE ERRORES) ---
volume = None
minVol, maxVol = -65.25, 0.0

def iniciar_audio():
    global volume, minVol, maxVol
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol_ptr = cast(interface, POINTER(IAudioEndpointVolume))
        vr = vol_ptr.GetVolumeRange()
        minVol, maxVol = vr[0], vr[1]
        print("✅ Audio de Windows conectado correctamente.")
        return vol_ptr
    except Exception as e:
        print(f"⚠️ Aviso: No se pudo controlar el volumen del sistema ({e})")
        print("El programa seguirá funcionando solo con la interfaz visual.")
        return None

volume = iniciar_audio()

# --- CONFIGURACIÓN MEDIAPIPE VISION ---
# Verifica que 'hand_landmarker.task' esté en: 
# D:\INGENIERIA CUATRIMESTRE 2\PROGRAMACION PARA ENTORNOS VIRTUALES\Eje_Uni3\
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"❌ ERROR: No se encuentra el archivo {model_path}")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# --- INICIO DE CAPTURA ---
cap = cv2.VideoCapture(0)
vol_per = 0

print("🎥 Cámara iniciada. Presiona 'q' para cerrar.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Procesamiento con MediaPipe Vision
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    resultados = detector.detect(mp_image)

    if resultados.hand_landmarks:
        for hand_landmarks in resultados.hand_landmarks:
            # Obtener puntos (0: Muñeca, 4: Pulgar, 8: Índice, 9: Medio Base, 12: Medio Punta)
            lm0, lm4, lm8, lm9 = hand_landmarks[0], hand_landmarks[4], hand_landmarks[8], hand_landmarks[9]
            lm12, lm13, lm16 = hand_landmarks[12], hand_landmarks[13], hand_landmarks[16]

            x0, y0 = int(lm0.x * w), int(lm0.y * h)
            x4, y4 = int(lm4.x * w), int(lm4.y * h)
            x8, y8 = int(lm8.x * w), int(lm8.y * h)
            x9, y9 = int(lm9.x * w), int(lm9.y * h)

            # Cálculos de distancia
            dist_ref = math.hypot(x9 - x0, y9 - y0)
            dist_dedos = math.hypot(x8 - x4, y8 - y4)

            # Mapeo a porcentaje (0-100)
            vol_per = np.interp(dist_dedos, [dist_ref * 0.3, dist_ref * 1.6], [0, 100])
            
            # Aplicar al sistema si el audio funciona
            if volume:
                try:
                    vol_db = np.interp(vol_per, [0, 100], [minVol, maxVol])
                    volume.SetMasterVolumeLevel(vol_db, None)
                    
                    # Gesto de Mute
                    if lm12.y > lm9.y and lm16.y > lm13.y:
                        volume.SetMute(1, None)
                        cv2.putText(frame, "SISTEMA MUTE", (w//2-80, 40), 1, 2, (0,0,255), 2)
                    else:
                        volume.SetMute(0, None)
                except:
                    pass

            # Dibujar elementos visuales
            cv2.circle(frame, (x4, y4), 10, (255, 0, 255), -1)
            cv2.circle(frame, (x8, y8), 10, (255, 0, 255), -1)
            cv2.line(frame, (x4, y4), (x8, y8), (255, 0, 255), 3)

    # --- INTERFAZ VISUAL ---
    vol_bar = np.interp(vol_per, [0, 100], [400, 150])
    # Dibujar barra de fondo
    cv2.rectangle(frame, (50, 150), (85, 400), (200, 200, 200), 3)
    # Dibujar barra de nivel
    cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)
    cv2.putText(frame, f'VOL: {int(vol_per)}%', (40, 450), 1, 2, (0, 255, 0), 2)

    cv2.imshow('Control IA - MediaPipe Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
detector.close()
cv2.destroyAllWindows()
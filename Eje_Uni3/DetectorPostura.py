import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Función para calcular el ángulo ---
def calcular_angulo(a, b, c):
    a = np.array(a) # Hombro
    b = np.array(b) # Cadera (vértice)
    c = np.array(c) # Rodilla
    
    radianes = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)
    
    if angulo > 180.0:
        angulo = 360 - angulo
    return angulo

# --- Configuración MediaPipe Vision ---
# Asegúrate de tener el archivo 'pose_landmarker_heavy.task' en la carpeta
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO # Optimizado para video
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- Variables de control de tiempo ---
inicio_mala_postura = None
tiempo_limite = 5  # Segundos
alerta_activa = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Espejo
    h, w, _ = frame.shape
    
    # Conversión a formato MediaPipe Vision
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # En modo VIDEO, necesitamos pasar el timestamp en milisegundos
    timestamp_ms = int(time.time() * 1000)
    resultados = detector.detect_for_video(mp_image, timestamp_ms)
    
    color_texto = (0, 255, 0) # Verde por defecto

    if resultados.pose_landmarks:
        # Extraemos el primer esqueleto detectado
        lm = resultados.pose_landmarks[0]
        
        # Puntos: Hombro (11), Cadera (23), Rodilla (25)
        # La nueva API devuelve objetos con .x y .y
        hombro = (int(lm[11].x * w), int(lm[11].y * h))
        cadera = (int(lm[23].x * w), int(lm[23].y * h))
        rodilla = (int(lm[25].x * w), int(lm[25].y * h))
        
        angulo = calcular_angulo(hombro, cadera, rodilla)
        
        # --- Lógica de la Postura ---
        es_mala_postura = angulo < 165 

        if es_mala_postura:
            if inicio_mala_postura is None:
                inicio_mala_postura = time.time()
            
            tiempo_transcurrido = time.time() - inicio_mala_postura
            color_texto = (0, 0, 255) # Rojo
            
            if tiempo_transcurrido >= tiempo_limite:
                alerta_activa = True
        else:
            inicio_mala_postura = None
            alerta_activa = False

        # --- Visualización ---
        # Dibujar líneas de referencia
        cv2.line(frame, hombro, cadera, (255, 255, 255), 2)
        cv2.line(frame, cadera, rodilla, (255, 255, 255), 2)
        
        # Dibujar puntos
        cv2.circle(frame, hombro, 5, (255, 0, 0), -1)
        cv2.circle(frame, cadera, 5, (255, 0, 0), -1)
        cv2.circle(frame, rodilla, 5, (255, 0, 0), -1)
        
        # Mostrar ángulo
        cv2.putText(frame, f"Angulo: {int(angulo)} deg", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)
        
        if alerta_activa:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10) # Marco rojo
            cv2.putText(frame, "!Sientate derecho!", (w//4, h//2), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow('Detector de Postura (Vision API)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
detector.close()
cv2.destroyAllWindows()
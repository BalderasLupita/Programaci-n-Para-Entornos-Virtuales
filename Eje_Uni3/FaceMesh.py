import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# --- CONFIGURACIÓN DE MEDIAPIPE VISION ---
# El archivo 'face_landmarker.task' debe estar en la misma carpeta
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

# Ajustamos los parámetros para evitar errores de compatibilidad
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1) # Quitamos la matriz de transformación para evitar el TypeError

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Presiona ESC para salir...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Preparar la imagen
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 2. Procesar
    results = detector.detect(mp_image)

    # 3. Dibujar los puntos (Landmarks)
    if results.face_landmarks:
        height, width, _ = frame.shape
        for face_landmarks in results.face_landmarks:
            for landmark in face_landmarks:
                # Convertir coordenadas normalizadas a píxeles
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Dibujar un punto verde pequeño
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # 4. Mostrar cámara (espejo)
    cv2.imshow('Face Mesh Vision API', cv2.flip(frame, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import os

# --- CONFIGURACIÓN DE RUTAS DINÁMICAS ---
# Obtiene la ruta de la carpeta donde está guardado este script (Eje_Uni3)
ruta_base = os.path.dirname(os.path.abspath(__file__))

# Define las rutas completas a los archivos del modelo
prototxt_path = os.path.join(ruta_base, "deploy.prototxt")
modelo_path = os.path.join(ruta_base, "res10_300x300_ssd_iter_140000.caffemodel")

# --- PASO 1: CONFIGURAR AMBOS DETECTORES ---

# Cargar Haar Cascade (usando la ruta interna de OpenCV)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar DNN (Caffe model) con manejo de error si los archivos no existen o están vacíos
try:
    red_dnn = cv2.dnn.readNetFromCaffe(prototxt_path, modelo_path)
except cv2.error as e:
    print(f"Error crítico: No se pudieron cargar los archivos del modelo DNN en: {ruta_base}")
    print("Asegúrate de que 'deploy.prototxt' y el '.caffemodel' no estén vacíos.")
    exit()

cap = cv2.VideoCapture(0)

# --- PASO 2: BUCLE DE DETECCIÓN COMPARATIVA ---

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_haar = frame.copy()
    frame_dnn = frame.copy()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- LÓGICA HAAR CASCADE ---
    inicio_haar = time.time()
    rostros_haar = haar_cascade.detectMultiScale(
        gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in rostros_haar:
        cv2.rectangle(frame_haar, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    fps_haar = 1 / (time.time() - inicio_haar + 1e-6)
    
    # --- LÓGICA DNN ---
    inicio_dnn = time.time()
    # Preprocesamiento para la red (Tamaño 300x300 y resta de media de colores)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    red_dnn.setInput(blob)
    detecciones = red_dnn.forward()
    
    h_frame, w_frame = frame.shape[:2]
    for i in range(detecciones.shape[2]):
        confianza = detecciones[0, 0, i, 2]
        if confianza > 0.5:  # Umbral de confianza del 50%
            # Extraer coordenadas y escalar al tamaño de la imagen
            box = detecciones[0, 0, i, 3:7] * np.array([w_frame, h_frame, w_frame, h_frame])
            (x1, y1, x2, y2) = box.astype("int")
            
            cv2.rectangle(frame_dnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_dnn, f"{confianza:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    fps_dnn = 1 / (time.time() - inicio_dnn + 1e-6)
    
    # --- MOSTRAR RESULTADOS ---
    cv2.putText(frame_haar, f"Haar FPS: {fps_haar:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame_dnn, f"DNN FPS: {fps_dnn:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Unir ambas imágenes para verlas en una sola ventana (opcional)
    # resultado = np.hstack((frame_haar, frame_dnn))
    # cv2.imshow('Comparativa: Izq(Haar) - Der(DNN)', resultado)
    
    cv2.imshow('Metodo Haar Cascade', frame_haar)
    cv2.imshow('Metodo DNN', frame_dnn)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
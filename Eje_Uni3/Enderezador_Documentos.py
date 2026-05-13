import cv2
import numpy as np

# Nombre de la ventana como constante para evitar errores de escritura
WIN_NAME = 'Corrector de Selfies'

def aplicar_correccion(imagen, intensidad):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, 1.3, 5)

    for (x, y, w, h) in rostros:
        pts_origen = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        
        # El offset simula el cambio de perspectiva focal
        offset = int((intensidad / 100.0) * (w * 0.15))
        
        pts_destino = np.float32([
            [x - offset, y], 
            [x + w + offset, y], 
            [x + w - offset, y + h], 
            [x + offset, y + h]
        ])

        M = cv2.getPerspectiveTransform(pts_origen, pts_destino)
        return cv2.warpPerspective(imagen, M, (imagen.shape[1], imagen.shape[0]))
    
    return imagen

# 1. Inicializar Cámara
cap = cv2.VideoCapture(0)

# 2. Crear Ventana y Trackbar ANTES del bucle
cv2.namedWindow(WIN_NAME)
cv2.createTrackbar('Efecto', WIN_NAME, 0, 100, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 3. Lectura segura del Trackbar
    try:
        val_intensidad = cv2.getTrackbarPos('Efecto', WIN_NAME)
    except cv2.error:
        val_intensidad = 0 # Valor por defecto si la ventana falla un segundo

    # 4. Procesamiento
    if val_intensidad > 0:
        resultado = aplicar_correccion(frame, val_intensidad)
    else:
        resultado = frame

    # 5. Mostrar imagen
    cv2.imshow(WIN_NAME, resultado)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
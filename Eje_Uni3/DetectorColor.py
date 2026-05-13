import cv2
import numpy as np
import os

def nada(x):
    pass

# Nombre del archivo para guardar la configuración
CONFIG_FILE = 'color_favorito.npy'

# 1. Configuración de Ventanas y Trackbars
cv2.namedWindow('Control')
cv2.createTrackbar('H Min', 'Control', 0, 179, nada)
cv2.createTrackbar('H Max', 'Control', 179, 179, nada)
cv2.createTrackbar('S Min', 'Control', 0, 255, nada)
cv2.createTrackbar('S Max', 'Control', 255, 255, nada)
cv2.createTrackbar('V Min', 'Control', 0, 255, nada)
cv2.createTrackbar('V Max', 'Control', 255, 255, nada)

def set_trackbars(hmin, hmax, smin, smax, vmin, vmax):
    cv2.setTrackbarPos('H Min', 'Control', hmin)
    cv2.setTrackbarPos('H Max', 'Control', hmax)
    cv2.setTrackbarPos('S Min', 'Control', smin)
    cv2.setTrackbarPos('S Max', 'Control', smax)
    cv2.setTrackbarPos('V Min', 'Control', vmin)
    cv2.setTrackbarPos('V Max', 'Control', vmax)

cap = cv2.VideoCapture(0)

print("--- DETECTOR HSV ---")
print("G: Guardar valores | C: Cargar valores | R: Restablecer | Q: Salir")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Pre-procesamiento
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Leer valores de trackbars
    h_min = cv2.getTrackbarPos('H Min', 'Control')
    h_max = cv2.getTrackbarPos('H Max', 'Control')
    s_min = cv2.getTrackbarPos('S Min', 'Control')
    s_max = cv2.getTrackbarPos('S Max', 'Control')
    v_min = cv2.getTrackbarPos('V Min', 'Control')
    v_max = cv2.getTrackbarPos('V Max', 'Control')
    
    # Crear máscara y limpiar ruido
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mascara = cv2.inRange(hsv, lower, upper)
    
    kernel = np.ones((5,5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    
    # Detección de contornos
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > 1000: # Ajustado a 1000 para evitar ruido pequeño
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "OBJETO DETECTADO", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resultado final
    resultado = cv2.bitwise_and(frame, frame, mask=mascara)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Mascara', mascara)
    cv2.imshow('Resultado', resultado)
    
    tecla = cv2.waitKey(1) & 0xFF
    
    # --- Lógica del Reto Personal ---
    if tecla == ord('g'): # GUARDAR
        valores = np.array([h_min, h_max, s_min, s_max, v_min, v_max])
        np.save(CONFIG_FILE, valores)
        print(f"✅ Valores guardados en {CONFIG_FILE}")

    elif tecla == ord('c'): # CARGAR
        if os.path.exists(CONFIG_FILE):
            v = np.load(CONFIG_FILE)
            set_trackbars(v[0], v[1], v[2], v[3], v[4], v[5])
            print("📂 Valores cargados correctamente.")
        else:
            print("❌ No hay archivo guardado.")

    elif tecla == ord('r'): # RESTABLECER
        set_trackbars(0, 179, 0, 255, 0, 255)
        print("🔄 Valores restablecidos.")

    elif tecla == ord('q') or tecla == 27:
        break

cap.release()
cv2.destroyAllWindows()
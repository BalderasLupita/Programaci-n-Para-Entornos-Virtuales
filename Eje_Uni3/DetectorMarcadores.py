import cv2
import cv2.aruco as aruco
import numpy as np
import math
from datetime import datetime
import os

# --- Configuración y Constantes ---
# ¡IMPORTANTE! Reemplaza este ID con tu ID de estudiante o número favorito
MI_ID_ESTUDIANTE = 24

# Parámetros de la cámara (valores aproximados)
# Una matriz de cámara más precisa mejorará la estimación de pose
matriz_camara = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coefs = np.zeros((4, 1))  # Asumimos sin distorsión por simplicidad

# Tamaño del marcador en metros (5 cm para marcadores pequeños de escritorio)
tamanio_marcador = 0.05

# Crear directorio para las fotos guardadas si no existe
directorio_fotos = "capturas_estudiante"
os.makedirs(directorio_fotos, exist_ok=True)

# Variables globales para efectos visuales y guardado
radio_circulo = 20
incremento_radio = 2
fotografia_guardada = False # Bandera para guardar solo una foto por aparición

# --- Funciones Auxiliares ---

def guardar_fotografia(frame, id_estudiante):
    """Guarda una foto del frame con la fecha y hora."""
    fecha_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{directorio_fotos}/deteccion_id_{id_estudiante}_{fecha_hora}.png"
    cv2.imwrite(nombre_archivo, frame)
    print(f"📸 ¡Foto guardada! {nombre_archivo}")

# --- Ciclo Principal del Detector ---

def detectar_y_personalizar():
    """Detecta marcadores, estima pose y personaliza para el ID del estudiante"""
    global radio_circulo, incremento_radio, fotografia_guardada

    # Configurar cámara
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Cargar diccionario y parámetros
    diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parametros = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(diccionario, parametros)

    print(f"🔍 Buscando marcadores... ID de estudiante: {MI_ID_ESTUDIANTE}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Detectar marcadores
        esquinas, ids, _ = detector.detectMarkers(frame)

        # Si hay marcadores, dibujarlos de forma básica primero
        if ids is not None:
            aruco.drawDetectedMarkers(frame, esquinas, ids)

            # Estimar pose para cada marcador
            for i in range(len(ids)):
                marker_id = ids[i][0]

                # --- 1. Determinar Puntos 3D y Resolver PnP (igual que en Paso 3) ---
                obj_points = np.array([[-tamanio_marcador/2, tamanio_marcador/2, 0],
                                      [tamanio_marcador/2, tamanio_marcador/2, 0],
                                      [tamanio_marcador/2, -tamanio_marcador/2, 0],
                                      [-tamanio_marcador/2, -tamanio_marcador/2, 0]], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_points, esquinas[i][0], matriz_camara, dist_coefs)

                if success:
                    # Dibujar ejes (igual que en Paso 3)
                    aruco.drawAxis(frame, matriz_camara, dist_coefs, rvec, tvec, 0.03)

                    # --- 2. Personalización Especial para MI_ID_ESTUDIANTE ---
                    if marker_id == MI_ID_ESTUDIANTE:
                        # a) Calcular centro del marcador en píxeles
                        centro_x = int(np.mean(esquinas[i][0][:, 0]))
                        centro_y = int(np.mean(esquinas[i][0][:, 1]))

                        # b) Efecto Visual Especial: Círculo Animado
                        radio_circulo += incremento_radio
                        if radio_circulo > 50 or radio_circulo < 20:
                            incremento_radio *= -1  # Invertir dirección

                        color_efecto = (255, 0, 255) # Magenta
                        cv2.circle(frame, (centro_x, centro_y), radio_circulo, color_efecto, 5)

                        # c) Mensaje Personalizado
                        cv2.putText(frame, f"¡ID {MI_ID_ESTUDIANTE} - BIENVENIDO!", 
                                    (centro_x - 150, centro_y - 120),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, color_efecto, 2)

                        # d) Guardar una sola foto por aparición
                        if not fotografia_guardada:
                            # Hacemos una copia del frame original para la foto antes de dibujar efectos
                            frame_para_guardar = cap.read()[1]
                            guardar_fotografia(frame_para_guardar, marker_id)
                            fotografia_guardada = True # Evitar guardar en el siguiente frame
                    
                    else:
                         # Reiniciar bandera para que pueda guardar foto de nuevo si reaparece
                         # (Solo si MI_ID_ESTUDIANTE no está visible)
                         if MI_ID_ESTUDIANTE not in ids:
                            fotografia_guardada = False


        cv2.imshow('Detector ArUco con Efectos Especiales', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ¡Genera primero el marcador! Si ya lo hiciste, puedes comentar esta línea
    # generar_un_marcador_especifico(MI_ID_ESTUDIANTE)

    detectar_y_personalizar()
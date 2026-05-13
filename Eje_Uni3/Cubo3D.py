import cv2
import cv2.aruco as aruco
import numpy as np

# --- Definición de la estructura 3D de la letra 'A' ---
# Definimos los puntos en el plano Z=0 y luego los replicamos en Z=0.03
def obtener_puntos_letra_A(l):
    # Puntos base de la 'A' (x, y, z)
    # Patas y punta
    puntos = np.float32([
        [-l/2, l/2, 0],   # 0: Pata izq abajo
        [0, -l/2, 0],     # 1: Punta arriba
        [l/2, l/2, 0],    # 2: Pata der abajo
        [-l/4, 0, 0],     # 3: Travesaño izq
        [l/4, 0, 0]       # 4: Travesaño der
    ])
    
    # Creamos la versión 3D (duplicamos puntos con profundidad z = l)
    puntos_3d = []
    for p in puntos: puntos_3d.append([p[0], p[1], 0])      # Capa fondo
    for p in puntos: puntos_3d.append([p[0], p[1], l/2])    # Capa frente
    
    # Conexiones (pares de índices de puntos)
    segmentos = [
        (0,1), (1,2), (3,4),       # Letra frontal
        (5,6), (6,7), (8,9),       # Letra trasera
        (0,5), (1,6), (2,7)        # Uniones de profundidad
    ]
    return np.array(puntos_3d, dtype=np.float32), segmentos

def main():
    cap = cv2.VideoCapture(0)
    matriz_camara = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
    dist_coefs = np.zeros((4, 1))
    
    tamanio_marcador = 0.05
    lado = 0.04
    puntos_letra, conexiones = obtener_puntos_letra_A(lado)
    
    diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parametros = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(diccionario, parametros)
    
    angulo = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        esquinas, ids, _ = detector.detectMarkers(frame)
        angulo = (angulo + 3) % 360 # Velocidad de rotación

        if ids is not None:
            for i in range(len(ids)):
                # 1. Estimar Pose
                obj_points = np.array([[-tamanio_marcador/2, tamanio_marcador/2, 0],
                                      [tamanio_marcador/2, tamanio_marcador/2, 0],
                                      [tamanio_marcador/2, -tamanio_marcador/2, 0],
                                      [-tamanio_marcador/2, -tamanio_marcador/2, 0]], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(obj_points, esquinas[i][0], matriz_camara, dist_coefs)
                
                if success:
                    # 2. Rotación suave (Eje Y)
                    R_extra, _ = cv2.Rodrigues(np.array([0, angulo * np.pi/180, 0], dtype=np.float32))
                    puntos_rotados = np.dot(puntos_letra, R_extra.T)

                    # 3. Proyectar puntos a 2D
                    imgpts, _ = cv2.projectPoints(puntos_rotados, rvec, tvec, matriz_camara, dist_coefs)
                    imgpts = np.int32(imgpts).reshape(-1, 2)

                    # 4. Color dinámico según distancia (tvec[2] es la profundidad Z)
                    distancia = np.linalg.norm(tvec)
                    # Si está cerca (<0.3m) es verde, si está lejos es rojo
                    color = (0, int(np.clip(255 - distancia*200, 0, 255)), int(np.clip(distancia*200, 0, 255)))

                    # 5. Dibujar los segmentos de la letra
                    for inicio, fin in conexiones:
                        cv2.line(frame, tuple(imgpts[inicio]), tuple(imgpts[fin]), color, 3)

        cv2.imshow('Letras 3D Dinámicas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
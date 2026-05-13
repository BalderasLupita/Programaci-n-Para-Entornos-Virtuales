import cv2
import cv2.aruco as aruco
import numpy as np

def sustituir_pantalla():
    # 1. Cargar la fuente de video que queremos proyectar
    video_proyectar = cv2.VideoCapture("mi_video.mp4") # Reemplaza con tu archivo
    cap_camara = cv2.VideoCapture(0)
    
    # Configuración de ArUco para tracking estable
    diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parametros = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(diccionario, parametros)

    print("📺 Buscando monitor... Presiona 'q' para salir.")

    while True:
        ret_cam, frame_cam = cap_camara.read()
        ret_vid, frame_vid = video_proyectar.read()
        
        if not ret_cam: break
        
        # Si el video proyectado termina, lo reiniciamos (Loop)
        if not ret_vid:
            video_proyectar.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_vid, frame_vid = video_proyectar.read()

        # 2. Detectar superficie (Usaremos un marcador ArUco como ancla)
        esquinas, ids, _ = detector.detectMarkers(frame_cam)
        
        if ids is not None:
            # Tomamos el primer marcador detectado
            pts_destino = esquinas[0][0] # 4 esquinas del marcador
            
            # --- Lógica de Proyección ---
            h_vid, w_vid = frame_vid.shape[:2]
            
            # Puntos de origen: las 4 esquinas del video original
            pts_origen = np.array([
                [0, 0], 
                [w_vid - 1, 0], 
                [w_vid - 1, h_vid - 1], 
                [0, h_vid - 1]
            ], dtype=np.float32)

            # 3. Calcular Homografía
            H, _ = cv2.findHomography(pts_origen, pts_destino)

            # 4. Warp: Transformar el frame del video al plano del monitor
            # Usamos el tamaño del frame de la cámara para la salida
            frame_warped = cv2.warpPerspective(frame_vid, H, (frame_cam.shape[1], frame_cam.shape[0]))

            # 5. Crear máscara para insertar el video en la cámara
            mascara = np.zeros((frame_cam.shape[0], frame_cam.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mascara, pts_destino.astype(int), 255)
            
            # Invertir máscara para "abrir un hueco" en la imagen de la cámara
            mascara_inv = cv2.bitwise_not(mascara)
            
            # Combinar: fondo de cámara con hueco + video transformado
            fondo = cv2.bitwise_and(frame_cam, frame_cam, mask=mascara_inv)
            video_final = cv2.bitwise_and(frame_warped, frame_warped, mask=mascara)
            
            frame_cam = cv2.add(fondo, video_final)

        cv2.imshow('AR Monitor Replacement', frame_cam)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_camara.release()
    video_proyectar.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sustituir_pantalla()
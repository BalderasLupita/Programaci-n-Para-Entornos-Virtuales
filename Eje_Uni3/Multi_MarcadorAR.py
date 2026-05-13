import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os

class MultiMarcadorAR:
    def __init__(self):
        # 1. Configurar detector ArUco (Versión actualizada para OpenCV 4.7+)
        self.diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parametros = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.diccionario, self.parametros)
        
        # 2. Base de datos de contenido
        self.base_datos = self.cargar_base_datos()
        
        # 3. Parámetros de cámara (Valores genéricos para Webcams estándar)
        self.matriz_camara = np.array([[1000, 0, 640],
                                       [0, 1000, 360],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coefs = np.zeros((4, 1))
        self.tamanio_marcador = 0.05  
        
        self.cache_contenido = {}

    def cargar_base_datos(self):
        if os.path.exists('contenido_marcadores.json'):
            with open('contenido_marcadores.json', 'r') as f:
                return json.load(f)
        
        # Diccionario inicial si no existe el JSON
        return {
            "0": {"tipo": "texto", "contenido": "HOLA MUNDO AR", "color": [0, 255, 0]},
            "1": {"tipo": "imagen", "contenido": "assets/logo.png"}, # Asegúrate de que exista
            "2": {"tipo": "video", "contenido": "assets/video.mp4", "loop": True}
        }

    def cargar_contenido(self, id_marcador):
        id_str = str(id_marcador)
        if id_str in self.cache_contenido:
            return self.cache_contenido[id_str]
        if id_str not in self.base_datos:
            return None
        
        info = self.base_datos[id_str]
        contenido = None

        if info["tipo"] == "imagen":
            if os.path.exists(info["contenido"]):
                contenido = cv2.imread(info["contenido"], cv2.IMREAD_UNCHANGED)
            else:
                # Generar un placeholder si no hay imagen
                contenido = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.putText(contenido, "IMG NOT FOUND", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        elif info["tipo"] == "video":
            if os.path.exists(info["contenido"]):
                contenido = {
                    'cap': cv2.VideoCapture(info["contenido"]),
                    'loop': info.get('loop', True)
                }
        
        elif info["tipo"] == "texto":
            # Creamos un lienzo para el texto
            img_texto = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(img_texto, info["contenido"], (20, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, info.get("color", [255, 255, 255]), 3)
            contenido = img_texto

        self.cache_contenido[id_str] = contenido
        return contenido

    def procesar_frame(self, frame):
        esquinas, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # Renderizar contenido
                contenido = self.cargar_contenido(marker_id)
                if contenido is not None:
                    frame = self.renderizar_contenido(frame, contenido, esquinas[i][0])
                
                # Info visual
                cv2.polylines(frame, [esquinas[i].astype(int)], True, (0, 255, 255), 2)
                cv2.putText(frame, f"ID: {marker_id}", (int(esquinas[i][0][0][0]), int(esquinas[i][0][0][1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return frame, ids

    def renderizar_contenido(self, frame, contenido, esquina):
        if isinstance(contenido, np.ndarray): # Imagen o Texto
            return self.superponer_imagen(frame, contenido, esquina)
        
        elif isinstance(contenido, dict) and 'cap' in contenido: # Video
            ret, frame_video = contenido['cap'].read()
            if not ret and contenido['loop']:
                contenido['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame_video = contenido['cap'].read()
            
            if ret:
                return self.superponer_imagen(frame, frame_video, esquina)
        return frame

    def superponer_imagen(self, frame, img_src, esquina):
        h_f, w_f = frame.shape[:2]
        h_s, w_s = img_src.shape[:2]
        
        pts_src = np.array([[0, 0], [w_s-1, 0], [w_s-1, h_s-1], [0, h_s-1]], dtype=np.float32)
        H, _ = cv2.findHomography(pts_src, esquina.astype(np.float32))
        
        # Deformar la imagen para que encaje en el marcador
        img_warped = cv2.warpPerspective(img_src, H, (w_f, h_f))
        
        # Crear máscara para la zona del marcador
        mask = np.zeros((h_f, w_f), dtype=np.uint8)
        cv2.fillConvexPoly(mask, esquina.astype(int), 255)
        
        # Si la imagen tiene canal alpha, usarlo; si no, usar la máscara del polígono
        if img_src.shape[2] == 4:
            # Separar canales y aplicar warp al alpha
            alpha_src = img_src[:,:,3]
            mask_alpha = cv2.warpPerspective(alpha_src, H, (w_f, h_f))
            mask = cv2.bitwise_and(mask, mask_alpha)

        # Invertir máscara
        mask_inv = cv2.bitwise_not(mask)
        
        # Combinar
        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(img_warped[:,:,:3], img_warped[:,:,:3], mask=mask)
        
        return cv2.add(bg, fg)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ar_system = MultiMarcadorAR()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame, ids = ar_system.procesar_frame(frame)
        cv2.imshow('Multi-Marcador AR', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    # Liberar recursos de video
    for c in ar_system.cache_contenido.values():
        if isinstance(c, dict) and 'cap' in c:
            c['cap'].release()
            
    cap.release()
    cv2.destroyAllWindows()
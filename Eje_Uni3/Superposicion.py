import cv2
import cv2.aruco as aruco
import numpy as np

def cargar_imagen_png(ruta, tamanio_deseado=None):
    """Carga imagen PNG manteniendo canal alpha o crea una por defecto"""
    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        # Generar imagen de respaldo si no existe el archivo
        img = np.zeros((300, 300, 4), dtype=np.uint8)
        cv2.rectangle(img, (10,10), (290,290), (0, 255, 0, 255), -1) # Fondo verde
        cv2.putText(img, "LOGO", (50, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255, 255), 5)
        print(f"⚠️ Nota: No se encontró '{ruta}', usando imagen generada.")
    
    if tamanio_deseado:
        img = cv2.resize(img, tamanio_deseado)
    return img

def superponer_imagen(frame, imagen_superponer, esquinas_destino):
    """Superpone una imagen con transparencia sobre las esquinas del marcador detectado"""
    h_img, w_img = imagen_superponer.shape[:2]
    
    # 1. Definir puntos de origen de la imagen (las 4 esquinas)
    pts_origen = np.array([[0, 0], [w_img-1, 0], 
                          [w_img-1, h_img-1], [0, h_img-1]], dtype=np.float32)
    
    # 2. Calcular la Homografía (matriz de transformación)
    H, _ = cv2.findHomography(pts_origen, esquinas_destino.astype(np.float32))
    
    # 3. Transformar la imagen para que coincida con la perspectiva del marcador
    img_warped = cv2.warpPerspective(imagen_superponer, H, (frame.shape[1], frame.shape[0]))
    
    # 4. Extraer máscara y canales
    # Si tiene canal alpha (PNG), lo usamos para la transparencia
    if imagen_superponer.shape[2] == 4:
        mascara = img_warped[:,:,3] / 255.0
    else:
        mascara = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        cv2.fillConvexPoly(mascara, esquinas_destino.astype(int), 1.0)

    # 5. Mezclar (Blending)
    # Convertimos la máscara a 3 canales para multiplicar directamente con el frame BGR
    mascara_3ch = cv2.merge([mascara, mascara, mascara])
    
    # Operación: (Fondo * (1 - máscara)) + (Imagen_AR * máscara)
    frame_bgr = frame.astype(float)
    img_warped_bgr = img_warped[:,:,:3].astype(float)
    
    resultado = frame_bgr * (1.0 - mascara_3ch) + img_warped_bgr * mascara_3ch
    
    return resultado.astype(np.uint8)

def main():
    cap = cv2.VideoCapture(0)
    
    # Configurar ArUco (Versiones nuevas de OpenCV)
    diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parametros = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(diccionario, parametros)
    
    imagen_ar = cargar_imagen_png("logo_ar.png")
    
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Detectar marcadores
        esquinas, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            # Puedes iterar por cada marcador detectado
            for i in range(len(ids)):
                frame = superponer_imagen(frame, imagen_ar, esquinas[i][0])
        
        cv2.imshow('Realidad Aumentada Básica', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
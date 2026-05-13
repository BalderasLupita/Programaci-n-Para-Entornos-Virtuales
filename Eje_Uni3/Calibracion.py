import cv2
import numpy as np
import os
import glob

# --- CONFIGURACIÓN GLOBAL ---
PATRON = (8, 5)              # Esquinas internas (para tablero de 9x6 cuadros)
TAMANIO_CUADRO = 0.025       # Tamaño del cuadro en metros (2.5 cm)
CARPETA_CALIB = "calibracion"
ARCHIVO_PARAMETROS = "parametros_camara.npz"

def crear_tablero_ajedrez(tamanio_pincel=30, num_x=9, num_y=6):
    """Paso 1: Genera la imagen del tablero para imprimir"""
    print("--- PASO 1: Generando tablero ---")
    ancho, alto = tamanio_pincel * num_x, tamanio_pincel * num_y
    tablero = np.ones((alto, ancho), dtype=np.uint8) * 255
    for i in range(num_y):
        for j in range(num_x):
            if (i + j) % 2 == 0:
                tablero[i*tamanio_pincel:(i+1)*tamanio_pincel, 
                        j*tamanio_pincel:(j+1)*tamanio_pincel] = 0
    
    tablero = cv2.copyMakeBorder(tablero, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite("tablero_calibracion.png", tablero)
    print("✅ Archivo 'tablero_calibracion.png' creado. ¡Imprímelo!")

def capturar_imagenes():
    """Paso 2: Captura fotos desde la webcam"""
    print("\n--- PASO 2: Captura de fotos ---")
    os.makedirs(CARPETA_CALIB, exist_ok=True)
    cap = cv2.VideoCapture(0)
    contador = 0
    
    print("Instrucciones:")
    print("- 'c': Capturar imagen (cuando veas líneas de colores)")
    print("- 'q': Terminar captura y empezar a calcular")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_encontrado, esquinas = cv2.findChessboardCorners(gris, PATRON, None)
        
        display = frame.copy()
        if ret_encontrado:
            cv2.drawChessboardCorners(display, PATRON, esquinas, ret_encontrado)
            cv2.putText(display, "LISTO PARA CAPTURAR (Presiona 'c')", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Captura de Calibracion', display)
        tecla = cv2.waitKey(1) & 0xFF
        
        if tecla == ord('c') and ret_encontrado:
            nombre = f"{CARPETA_CALIB}/img_{contador:03d}.png"
            cv2.imwrite(nombre, frame)
            print(f"✅ Guardada: {nombre}")
            contador += 1
        elif tecla == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def procesar_calibracion():
    """Paso 3: Calcula la matriz y guarda resultados"""
    print("\n--- PASO 3: Calculando parámetros ---")
    puntos_3d_base = np.zeros((PATRON[0] * PATRON[1], 3), np.float32)
    puntos_3d_base[:, :2] = np.mgrid[0:PATRON[0], 0:PATRON[1]].T.reshape(-1, 2)
    puntos_3d_base *= TAMANIO_CUADRO

    lista_3d, lista_2d = [], []
    imagenes = glob.glob(f"{CARPETA_CALIB}/*.png")

    for img_path in imagenes:
        img = cv2.imread(img_path)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, esquinas = cv2.findChessboardCorners(gris, PATRON, None)
        
        if ret:
            criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            esquinas_sub = cv2.cornerSubPix(gris, esquinas, (11,11), (-1,-1), criterios)
            lista_3d.append(puntos_3d_base)
            lista_2d.append(esquinas_sub)

    if not lista_2d:
        print("❌ Error: No se detectaron tableros en las fotos guardadas.")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(lista_3d, lista_2d, gris.shape[::-1], None, None)
    
    np.savez(ARCHIVO_PARAMETROS, matriz_camara=mtx, dist_coefs=dist)
    print(f"✅ Calibración exitosa. Error: {ret:.4f}")
    print(f"💾 Parámetros guardados en {ARCHIVO_PARAMETROS}")
    return mtx, dist

def demostracion_en_vivo(mtx, dist):
    """Muestra la cámara corregida en tiempo real"""
    print("\n--- DEMOSTRACIÓN: Comparativa en vivo ---")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        nueva_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        corregido = cv2.undistort(frame, mtx, dist, None, nueva_mtx)
        
        # Recorte opcional para limpiar bordes negros
        x, y, w_roi, h_roi = roi
        corregido = corregido[y:y+h_roi, x:x+w_roi]
        corregido = cv2.resize(corregido, (w, h)) # Redimensionar para comparar

        comparacion = np.hstack([frame, corregido])
        cv2.putText(comparacion, "ORIGINAL", (10, 30), 1, 2, (0,0,255), 2)
        cv2.putText(comparacion, "CORREGIDA", (w + 10, 30), 1, 2, (0,255,0), 2)
        
        cv2.imshow('Calibracion Realizada', comparacion)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 1. Crear tablero (puedes comentar esto después de imprimirlo)
    crear_tablero_ajedrez()
    
    input("\nPresiona Enter cuando ya tengas el tablero impreso para empezar a capturar...")
    
    # 2. Capturar fotos
    capturar_imagenes()
    
    # 3. Procesar y Guardar
    mtx, dist = procesar_calibracion()
    
    # 4. Ver resultado
    if mtx is not None:
        demostracion_en_vivo(mtx, dist)
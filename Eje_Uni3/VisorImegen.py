import cv2
import numpy as np

def mostrar_valor_pixel(evento, x, y, flags, param):
    """Callback para mostrar info del píxel"""
    # Desempaquetamos la lista que pasamos en el param
    img_actual, img_original = param
    
    if evento == cv2.EVENT_MOUSEMOVE:
        img_temp = img_actual.copy()
        if 0 <= y < img_temp.shape[0] and 0 <= x < img_temp.shape[1]:
            # Extraemos color de la ORIGINAL para que siempre sea el real
            b, g, r = img_original[y, x]
            texto = f"B:{b} G:{g} R:{r} | ({x},{y})"
            
            # Dibujamos un pequeño fondo para el texto
            cv2.rectangle(img_temp, (5, 5), (280, 35), (0, 0, 0), -1)
            cv2.putText(img_temp, texto, (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(img_temp, (x, y), 5, (0, 255, 0), 1)
            
            cv2.imshow('Desmenuzador', img_temp)

def main():
    # 1. Cargar o crear imagen
    img_original = cv2.imread('test.jpg')
    if img_original is None:
        img_original = np.zeros((400, 600, 3), dtype=np.uint8)
        for i in range(400):
            for j in range(600):
                img_original[i, j] = [j % 256, i % 256, (i+j) % 256]

    # 2. Inicializar la ventana ANTES de cualquier otra cosa
    cv2.namedWindow('Desmenuzador', cv2.WINDOW_AUTOSIZE)
    
    # 3. Estado inicial
    img_mostrar = img_original.copy()
    
    print("--- CONTROLADOR DE CANALES ---")
    print("1: Azul | 2: Verde | 3: Rojo | 4: Original | Q: Salir")

    while True:
        # ACTUALIZAMOS EL CALLBACK en cada ciclo para que use la imagen actual
        cv2.setMouseCallback('Desmenuzador', mostrar_valor_pixel, [img_mostrar, img_original])
        cv2.imshow('Desmenuzador', img_mostrar)
        
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('1'): # CANAL AZUL
            img_mostrar = np.zeros_like(img_original)
            img_mostrar[:, :, 0] = img_original[:, :, 0]
        elif tecla == ord('2'): # CANAL VERDE
            img_mostrar = np.zeros_like(img_original)
            img_mostrar[:, :, 1] = img_original[:, :, 1]
        elif tecla == ord('3'): # CANAL ROJO
            img_mostrar = np.zeros_like(img_original)
            img_mostrar[:, :, 2] = img_original[:, :, 2]
        elif tecla == ord('4'): # ORIGINAL
            img_mostrar = img_original.copy()
        elif tecla == ord('q') or tecla == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
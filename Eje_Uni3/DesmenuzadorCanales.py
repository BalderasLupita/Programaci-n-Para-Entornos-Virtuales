import cv2
import numpy as np

def desmenuzador_canales():
    imagen = cv2.imread('test.jpg')
    if imagen is None:
        imagen = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(imagen, "Imagen de Prueba", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    while True:
        cv2.imshow('Canal', imagen)
        tecla = cv2.waitKey(0) & 0xFF
        
        # Crear base negra
        resultado = np.zeros_like(imagen)
        
        if tecla == ord('1'): # Azul
            resultado[:, :, 0] = imagen[:, :, 0]
            cv2.imshow('Canal', resultado)
        elif tecla == ord('2'): # Verde
            resultado[:, :, 1] = imagen[:, :, 1]
            cv2.imshow('Canal', resultado)
        elif tecla == ord('3'): # Rojo
            resultado[:, :, 2] = imagen[:, :, 2]
            cv2.imshow('Canal', resultado)
        elif tecla == ord('4'): # Completa
            cv2.imshow('Canal', imagen)
        elif tecla == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    desmenuzador_canales()
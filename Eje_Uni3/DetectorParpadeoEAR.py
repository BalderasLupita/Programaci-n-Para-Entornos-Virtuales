# Dentro del bucle principal del detector de color:
import numpy as np

def calcular_ear(puntos_ojo):
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)[cite: 1]
    # Usando índices relativos de la lista de puntos del ojo
    p1, p2, p3, p4, p5, p6 = puntos_ojo[0], puntos_ojo[1], puntos_ojo[2], puntos_ojo[3], puntos_ojo[4], puntos_ojo[5]
    
    A = np.linalg.norm(np.array(p2) - np.array(p6))
    B = np.linalg.norm(np.array(p3) - np.array(p5))
    C = np.linalg.norm(np.array(p1) - np.array(p4))
    
    return (A + B) / (2.0 * C)

# Umbral sugerido: 0.2[cite: 1]
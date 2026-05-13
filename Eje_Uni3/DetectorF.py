import cv2
import numpy as np

def detectar_formas(contorno):
    """
    Analiza el contorno para determinar la forma geométrica.
    """
    # Calculamos el perímetro y aproximamos el polígono
    # Usamos un factor de 0.03 para un equilibrio entre precisión y ruido
    peri = cv2.arcLength(contorno, True)
    aproximacion = cv2.approxPolyDP(contorno, 0.03 * peri, True)
    
    vertices = len(aproximacion)
    
    if vertices == 3:
        return "Triangulo"
    elif vertices == 4:
        # Diferenciar entre Cuadrado y Rectángulo usando la relación de aspecto
        x, y, w, h = cv2.boundingRect(aproximacion)
        relacion_aspecto = float(w) / h
        if 0.90 <= relacion_aspecto <= 1.10:
            return "Cuadrado"
        else:
            return "Rectangulo"
    elif vertices == 5:
        return "Pentagono"
    elif vertices > 8:
        # Para círculos, comparamos el área del contorno con la de un círculo perfecto
        area = cv2.contourArea(contorno)
        (x, y), radio = cv2.minEnclosingCircle(contorno)
        area_ideal = np.pi * (radio ** 2)
        if abs(area - area_ideal) / area_ideal < 0.2:
            return "Circulo"
            
    return "Desconocido"

# Inicializar cámara
cap = cv2.VideoCapture(0)

print("Controles: 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reiniciamos el contador en cada frame
    conteo = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0, "Pentagono": 0}
    
    # --- 1. PREPROCESAMIENTO ---
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # El desenfoque elimina el ruido que Canny confunde con bordes
    desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # --- 2. DETECCIÓN DE BORDES MEJORADA ---
    # Umbrales más bajos para detectar figuras con menos contraste
    bordes = cv2.Canny(desenfoque, 40, 120)
    
    # Operación Morfológica: 'Cierra' los bordes que Canny dejó abiertos
    kernel = np.ones((3,3), np.uint8)
    bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)

    # --- 3. BÚSQUEDA Y FILTRADO DE CONTORNOS ---
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contornos:
        area = cv2.contourArea(c)
        
        # Ignorar figuras muy pequeñas (ruido)
        if area < 800:
            continue
            
        forma = detectar_formas(c)
        
        if forma != "Desconocido":
            conteo[forma] += 1
            
            # Dibujar el contorno y el nombre
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            
            # Calcular centro para poner el texto
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, forma, (cX - 30, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- 4. INTERFAZ DE USUARIO (DASHBOARD) ---
    # Fondo para el contador
    cv2.rectangle(frame, (0, 0), (640, 45), (0, 0, 0), -1)
    resumen = f"T: {conteo['Triangulo']} | C: {conteo['Cuadrado']} | R: {conteo['Rectangulo']} | O: {conteo['Circulo']}"
    cv2.putText(frame, resumen, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Mostrar ventanas
    cv2.imshow('Detector de Formas Pro', frame)
    cv2.imshow('Mascara de Bordes (Debug)', bordes)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
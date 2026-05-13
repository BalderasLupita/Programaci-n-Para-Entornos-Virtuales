import cv2
import numpy as np

def detectar_formas(contorno):
    peri = cv2.arcLength(contorno, True)
  
    aproximacion = cv2.approxPolyDP(contorno, 0.04 * peri, True)
    vertices = len(aproximacion)
    
    if vertices == 3:
        return "Triangulo"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(aproximacion)
        aspect_ratio = w / float(h)
        return "Cuadrado" if 0.95 <= aspect_ratio <= 1.05 else "Rectangulo"
    elif vertices == 5:
        return "Pentagono"
    elif vertices > 6:
        return "Circulo"
    return "Desconocido"

cap = cv2.VideoCapture(0)


ultimo_conteo_total = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
 
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    desenfoque = cv2.GaussianBlur(gris, (7, 7), 0) # Un poco más de desenfoque ayuda
    bordes = cv2.Canny(desenfoque, 50, 150)
  
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
   
    conteo_actual = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0, "Pentagono": 0}
    
    for contorno in contornos:
        if cv2.contourArea(contorno) < 1000: # Filtro de tamaño
            continue
            
        forma = detectar_formas(contorno)
        
        if forma in conteo_actual:
            conteo_actual[forma] += 1
            
            
            cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 2)
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, forma, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  
    total_objetos = sum(conteo_actual.values())
    if total_objetos > ultimo_conteo_total:
        print('\a') # Beep!
        print(conteo_actual)
    ultimo_conteo_total = total_objetos

    
    cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
    
    info = f"Tri: {conteo_actual['Triangulo']} | Cua: {conteo_actual['Cuadrado']} | Rec: {conteo_actual['Rectangulo']} | Cir: {conteo_actual['Circulo']}"
    cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Contador Pro Real-Time', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
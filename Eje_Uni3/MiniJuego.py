import cv2
import mediapipe as mp
import numpy as np
import random
import time
import json
import os
import sys
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QPushButton, QWidget, QHBoxLayout)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

# --- CLASE BASE: MOTOR DE VISIÓN Y MECÁNICA ---
class JuegoARBase:
    def __init__(self):
        # Configuración del detector de manos (API Vision)
        modelo_path = 'hand_landmarker.task'
        if not os.path.exists(modelo_path):
            print(f"ERROR: No se encuentra {modelo_path}")
            sys.exit()

        base_options = python.BaseOptions(model_asset_path=modelo_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Parámetros del juego
        self.ancho, self.alto = 1280, 720
        self.puntos = 0
        self.vidas = 3
        self.objetos = []
        self.velocidad_base = 6
        self.ultima_posicion = (0, 0)
        
        self.colores = {
            'bueno': (0, 255, 0),    # Verde
            'malo': (0, 0, 255),     # Rojo
            'especial': (255, 255, 0) # Amarillo
        }

    def crear_objeto(self):
        tipo = random.choice(['bueno', 'malo', 'especial'])
        radio = 30 if tipo == 'especial' else 22
        objeto = {
            'x': random.randint(50, self.ancho - 50),
            'y': -50,
            'tipo': tipo,
            'radio': radio,
            'puntos': 10 if tipo == 'bueno' else -15 if tipo == 'malo' else 50,
            'velocidad': self.velocidad_base * (1.4 if tipo == 'malo' else 1.0)
        }
        self.objetos.append(objeto)

    def actualizar_objetos(self):
        objetos_activos = []
        for obj in self.objetos:
            obj['y'] += obj['velocidad']
            if obj['y'] < self.alto:
                objetos_activos.append(obj)
            elif obj['tipo'] == 'bueno':
                self.puntos = max(0, self.puntos - 5) # Penalizar si se escapan
        
        self.objetos = objetos_activos
        if len(self.objetos) < 7 and random.random() < 0.05:
            self.crear_objeto()

    def detectar_mano(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)
        
        posicion_mano = None
        if res.hand_landmarks:
            landmarks = res.hand_landmarks[0]
            # Usar la base de la palma (punto 0) para detectar colisión
            palma = landmarks[0]
            posicion_mano = (int(palma.x * self.ancho), int(palma.y * self.alto))
            self.ultima_posicion = posicion_mano
            
            # Dibujar esqueleto de la mano
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x*self.ancho), int(lm.y*self.alto)), 5, (255, 0, 255), -1)
        
        return frame, posicion_mano, res

    def verificar_colisiones(self, pos_mano):
        if not pos_mano: return
        mx, my = pos_mano
        nuevos_objetos = []
        for obj in self.objetos:
            dist = np.sqrt((mx - obj['x'])**2 + (my - obj['y'])**2)
            if dist < obj['radio'] + 40:
                self.puntos += obj['puntos']
                if obj['tipo'] == 'malo': self.vidas -= 1
            else:
                nuevos_objetos.append(obj)
        self.objetos = nuevos_objetos

# --- SEMANA 2 & 3: GESTOS Y NIVELES ---
class JuegoCompleto(JuegoARBase):
    def __init__(self):
        super().__init__()
        self.nivel = 1
        self.ranking_file = 'ranking.json'

    def detectar_gesto(self, res):
        if not res.hand_landmarks: return "Ninguno"
        hand = res.hand_landmarks[0]
        # Lógica simplificada de dedos (y de la punta < y del nudillo)
        dedos = [hand[i].y < hand[i-2].y for i in [8, 12, 16, 20]]
        cont = dedos.count(True)
        
        if cont == 0: return "Puño (Frenado)"
        if cont == 2: return "Paz (Bonus)"
        if cont >= 4: return "Abierta (Imán)"
        return "Normal"

    def aplicar_gesto(self, gesto):
        if "Puño" in gesto:
            for o in self.objetos: o['velocidad'] = 2
        elif "Abierta" in gesto:
            for o in self.objetos:
                o['x'] += (self.ultima_posicion[0] - o['x']) * 0.1
                o['y'] += (self.ultima_posicion[1] - o['y']) * 0.1

    def actualizar_nivel(self):
        nuevo_nivel = (self.puntos // 100) + 1
        if nuevo_nivel > self.nivel:
            self.nivel = nuevo_nivel
            self.velocidad_base += 1

# --- SEMANA 4: APLICACIÓN PYQT6 ---
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AR Catcher Pro - MediaPipe Vision")
        self.setFixedSize(1280, 800)

        self.motor = JuegoCompleto()
        
        # UI Layout
        self.label_video = QLabel()
        self.btn_reiniciar = QPushButton("REINICIAR JUEGO")
        self.btn_reiniciar.clicked.connect(self.reiniciar)
        self.btn_reiniciar.setFixedHeight(40)

        layout = QVBoxLayout()
        layout.addWidget(self.label_video)
        layout.addWidget(self.btn_reiniciar)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game)
        self.timer.start(30)

    def reiniciar(self):
        self.motor = JuegoCompleto()

    def update_game(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))
        
        # Procesamiento
        frame, pos, res = self.motor.detectar_mano(frame)
        gesto = self.motor.detectar_gesto(res)
        self.motor.aplicar_gesto(gesto)
        self.motor.actualizar_objetos()
        self.motor.verificar_colisiones(pos)
        self.motor.actualizar_nivel()
        
        # Dibujar UI en el frame
        self.motor.dibujar_elementos(frame, gesto)
        
        # Convertir a QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img_qt = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(img_qt))

        if self.motor.vidas <= 0:
            self.timer.stop()
            print(f"GAME OVER - Puntos: {self.motor.puntos}")
            self.reiniciar()
            self.timer.start(30)

    def closeEvent(self, event):
        self.cap.release()

# Añadimos un método de dibujo al motor para limpiar el update_game
def dibujar_elementos(self, frame, gesto):
    for obj in self.objetos:
        cv2.circle(frame, (obj['x'], int(obj['y'])), obj['radio'], self.colores[obj['tipo']], -1)
    
    # Texto de información
    info = f"Puntos: {self.puntos} | Vidas: {self.vidas} | Nivel: {self.nivel}"
    cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Gesto: {gesto}", (900, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

# Inyectar el método de dibujo dinámicamente
JuegoCompleto.dibujar_elementos = dibujar_elementos

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VentanaPrincipal()
    window.show()
    sys.exit(app.exec())
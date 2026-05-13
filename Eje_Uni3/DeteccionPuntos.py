import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QListWidget, QListWidgetItem, QSlider, QCheckBox,
                             QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# ==========================================
# SEMANA 1: MOTOR DE DETECCIÓN FACIAL
# ==========================================
class DetectorFacial:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.indices = {
            "ojo_izq_centro": 468, "ojo_der_centro": 473,
            "ojo_izq_externo": 33, "ojo_izq_interno": 133,
            "ojo_der_externo": 362, "ojo_der_interno": 263,
            "ceja_izq_centro": 70, "ceja_der_centro": 336,
            "nariz_punta": 1, "boca_izq": 61, "boca_der": 291,
            "boca_sup": 0, "menton": 152, "frente_izq": 10, "frente_der": 338
        }

    def detectar(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = self.face_mesh.process(rgb)
        puntos = {}
        h, w = frame.shape[:2]
        
        if resultados.multi_face_landmarks:
            landmarks = resultados.multi_face_landmarks[0].landmark
            for nombre, idx in self.indices.items():
                puntos[nombre] = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            return puntos
        return None

# ==========================================
# SEMANA 2 & 3: FILTROS ESTÁTICOS Y ANIMADOS
# ==========================================
class MotorFiltros(DetectorFacial):
    def __init__(self):
        super().__init__()
        self.tiempo_inicio = time.time()
        self.particulas = []
        self.escala_filtro = 1.0
        self.filtros_img = self._generar_assets_basicos()

    def _generar_assets_basicos(self):
        # Crear imágenes simples para filtros si no hay archivos externos
        assets = {}
        # Gafas base
        gafas = np.zeros((100, 300, 4), dtype=np.uint8)
        cv2.rectangle(gafas, (10, 30), (130, 70), (0, 0, 0, 255), -1)
        cv2.rectangle(gafas, (170, 30), (290, 70), (0, 0, 0, 255), -1)
        cv2.line(gafas, (130, 50), (170, 50), (0, 0, 0, 255), 5)
        assets["gafas"] = gafas
        
        # Bigote base
        bigote = np.zeros((60, 160, 4), dtype=np.uint8)
        cv2.ellipse(bigote, (50, 30), (40, 20), 0, 0, 360, (20, 20, 20, 255), -1)
        cv2.ellipse(bigote, (110, 30), (40, 20), 0, 0, 360, (20, 20, 20, 255), -1)
        assets["bigote"] = bigote
        return assets

    def superponer_transparente(self, fondo, overlay, pos):
        x, y = pos
        h, w = overlay.shape[:2]
        if y < 0 or x < 0 or y+h > fondo.shape[0] or x+w > fondo.shape[1]:
            return fondo
        
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            fondo[y:y+h, x:x+w, c] = fondo[y:y+h, x:x+w, c] * (1 - alpha) + overlay[:, :, c] * alpha
        return fondo

    def aplicar_filtro(self, frame, puntos, tipo):
        if not puntos: return frame
        
        if tipo == "gafas":
            izq, der = puntos["ojo_izq_externo"], puntos["ojo_der_externo"]
            ancho = int(abs(der[0] - izq[0]) * 2.5 * self.escala_filtro)
            alto = int(ancho * 0.3)
            img = cv2.resize(self.filtros_img["gafas"], (ancho, alto))
            pos = (int((izq[0]+der[0])/2 - ancho/2), int((izq[1]+der[1])/2 - alto/2))
            frame = self.superponer_transparente(frame, img, pos)

        elif tipo == "bigote":
            nariz, boca = puntos["nariz_punta"], puntos["boca_sup"]
            ancho = int(abs(puntos["boca_der"][0] - puntos["boca_izq"][0]) * 1.5 * self.escala_filtro)
            alto = int(ancho * 0.4)
            img = cv2.resize(self.filtros_img["bigote"], (ancho, alto))
            pos = (nariz[0] - ancho//2, (nariz[1] + boca[1])//2 - alto//2)
            frame = self.superponer_transparente(frame, img, pos)

        elif tipo == "particulas":
            self._actualizar_particulas(puntos["nariz_punta"])
            for p in self.particulas:
                cv2.circle(frame, (int(p['x']), int(p['y'])), p['r'], p['c'], -1)
                
        return frame

    def _actualizar_particulas(self, centro):
        if len(self.particulas) < 20:
            self.particulas.append({
                'x': centro[0], 'y': centro[1], 
                'vx': np.random.uniform(-3, 3), 'vy': np.random.uniform(-3, 3),
                'r': np.random.randint(2, 6), 'c': (0, 255, 255), 'v': 20
            })
        for p in self.particulas[:]:
            p['x'] += p['vx']; p['y'] += p['vy']; p['v'] -= 1
            if p['v'] <= 0: self.particulas.remove(p)

# ==========================================
# SEMANA 4: INTERFAZ PYQT6
# ==========================================
class AppFiltros(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SnapAR Pro - Capítulo 13")
        self.motor = MotorFiltros()
        self.filtro_actual = "gafas"
        self.cap = cv2.VideoCapture(0)
        
        self._init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        
        # Area de Video
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background: black; border: 2px solid #333;")
        layout.addWidget(self.video_label)
        
        # Panel de Control
        controls = QVBoxLayout()
        
        # Lista de Filtros
        self.lista = QListWidget()
        self.lista.addItems(["gafas", "bigote", "particulas", "ninguno"])
        self.lista.setCurrentRow(0)
        self.lista.currentRowChanged.connect(self.cambiar_filtro)
        controls.addWidget(QLabel("Seleccionar Filtro:"))
        controls.addWidget(self.lista)
        
        # Slider Tamaño
        controls.addWidget(QLabel("Tamaño del Filtro:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(5, 20)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self.cambiar_escala)
        controls.addWidget(self.slider)
        
        # Botón Captura
        btn_snap = QPushButton("📸 Tomar Foto")
        btn_snap.clicked.connect(self.take_snapshot)
        controls.addWidget(btn_snap)
        
        layout.addLayout(controls)
        self.setCentralWidget(main_widget)

    def cambiar_filtro(self, i):
        self.filtro_actual = self.lista.item(i).text()

    def cambiar_escala(self, v):
        self.motor.escala_filtro = v / 10.0

    def take_snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            puntos = self.motor.detectar(frame)
            frame = self.motor.aplicar_filtro(frame, puntos, self.filtro_actual)
            cv2.imwrite(f"snapshot_{int(time.time())}.jpg", frame)
            QMessageBox.information(self, "Guardado", "¡Foto guardada con éxito!")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            puntos = self.motor.detectar(frame)
            frame = self.motor.aplicar_filtro(frame, puntos, self.filtro_actual)
            
            # Convertir para Qt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppFiltros()
    window.show()
    sys.exit(app.exec())
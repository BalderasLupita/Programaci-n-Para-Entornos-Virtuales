import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
import time
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGroupBox, QListWidget, QSlider)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class DetectorFacial:
    def __init__(self):
        try:
            base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
            # Configuración simplificada para máxima compatibilidad
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=1)
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error crítico: {e}")
            sys.exit()
        
        self.indices = {
            "ojo_izq_centro": 468, "ojo_der_centro": 473,
            "ojo_izq_externo": 33, "ojo_der_externo": 362,
            "frente_izq": 10, "frente_der": 338,
            "nariz_punta": 1, "boca_sup": 0, "boca_izq": 61, "boca_der": 291
        }

    def detectar(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        resultado = self.detector.detect(mp_image)
        
        puntos = {}
        h, w = frame.shape[:2]
        if resultado.face_landmarks:
            landmarks = resultado.face_landmarks[0]
            for nombre, idx in self.indices.items():
                puntos[nombre] = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            return puntos
        return None

class MotorFiltros(DetectorFacial):
    def __init__(self):
        super().__init__()
        self.filtros_img = self._generar_recursos()

    def _generar_recursos(self):
        filtros = {}
        gafas = np.zeros((100, 200, 4), dtype=np.uint8)
        cv2.rectangle(gafas, (10, 30), (85, 70), (20, 20, 20, 255), -1)
        cv2.rectangle(gafas, (115, 30), (190, 70), (20, 20, 20, 255), -1)
        cv2.line(gafas, (85, 50), (115, 50), (20, 20, 20, 255), 8)
        filtros["gafas"] = gafas
        return filtros

    def superponer(self, fondo, overlay, pos):
        x, y = pos
        h, w = overlay.shape[:2]
        if y < 0 or x < 0 or y+h > fondo.shape[0] or x+w > fondo.shape[1]:
            return fondo
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            fondo[y:y+h, x:x+w, c] = fondo[y:y+h, x:x+w, c] * (1 - alpha) + overlay[:, :, c] * alpha
        return fondo

class SnapARApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SnapAR Studio - Vision V2")
        self.setMinimumSize(1000, 600)
        self.motor = MotorFiltros()
        self.cap = cv2.VideoCapture(0)
        self.filtro_activo = "Ninguno"
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, 7)
        panel = QVBoxLayout()
        self.lista = QListWidget()
        self.lista.addItems(["Ninguno", "Gafas", "Efecto Neon"])
        self.lista.setCurrentRow(0)
        self.lista.currentRowChanged.connect(self.cambiar_filtro)
        panel.addWidget(QLabel("FILTROS"))
        panel.addWidget(self.lista)
        layout.addLayout(panel, 3)
        self.setCentralWidget(main_widget)

    def cambiar_filtro(self, index):
        self.filtro_activo = self.lista.item(index).text()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        puntos = self.motor.detectar(frame)
        
        if puntos:
            if self.filtro_activo == "Gafas":
                img = self.motor.filtros_img["gafas"]
                ancho = int(abs(puntos["ojo_der_externo"][0] - puntos["ojo_izq_externo"][0]) * 2.2)
                alto = int(ancho * 0.5)
                img = cv2.resize(img, (ancho, alto))
                pos = (puntos["ojo_izq_centro"][0] - int(ancho*0.25), puntos["ojo_izq_centro"][1] - int(alto*0.5))
                frame = self.motor.superponer(frame, img, pos)
            elif self.filtro_activo == "Efecto Neon":
                t = time.time()
                color = (int(127+127*math.sin(t*3)), 255, 0)
                cv2.circle(frame, puntos["ojo_izq_centro"], 10, color, -1)
                cv2.circle(frame, puntos["ojo_der_centro"], 10, color, -1)

        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnapARApp()
    window.show()
    sys.exit(app.exec())
import cv2
import mediapipe as mp
import numpy as np
import sys
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

# --- CLASE DE VISIÓN (PROCESAMIENTO) ---
class MotorVision:
    def __init__(self):
        # El archivo .task debe estar en la misma carpeta
        modelo_path = 'face_landmarker.task'
        if not os.path.exists(modelo_path):
            print(f"ERROR: No se encuentra el archivo {modelo_path}")
            sys.exit()

        base_options = python.BaseOptions(model_asset_path=modelo_path)
        
        # Hemos simplificado las opciones eliminando la matriz de transformación 
        # que causaba el error de TypeError.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def procesar_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect(mp_image)

    def dibujar_accesorio(self, frame, resultado, tipo="gafas"):
        if not resultado or not resultado.face_landmarks:
            return frame

        h, w, _ = frame.shape
        puntos = resultado.face_landmarks[0]

        if tipo == "gafas":
            # Puntos 33 y 263 (esquinas exteriores de los ojos)
            p1 = puntos[33]
            p2 = puntos[263]
            
            cx = int((p1.x + p2.x) / 2 * w)
            cy = int((p1.y + p2.y) / 2 * h)
            ancho = int(abs(p2.x - p1.x) * w * 1.6)
            
            # Dibujar gafas sencillas
            cv2.ellipse(frame, (cx, cy), (ancho // 2, 25), 0, 0, 360, (30, 30, 30), -1)
            cv2.circle(frame, (int(p1.x * w), int(p1.y * h)), 15, (220, 220, 220), 2)
            cv2.circle(frame, (int(p2.x * w), int(p2.y * h)), 15, (220, 220, 220), 2)

        elif tipo == "sombrero":
            # Punto 10: Tope de la frente
            frente = puntos[10]
            fx, fy = int(frente.x * w), int(frente.y * h)
            
            cv2.rectangle(frame, (fx - 70, fy - 100), (fx + 70, fy - 40), (0, 0, 200), -1)
            cv2.line(frame, (fx - 110, fy - 40), (fx + 110, fy - 40), (0, 0, 150), 6)

        return frame

# --- CLASE DE INTERFAZ (PYQT6) ---
class ProbadorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Probador Virtual")
        self.setFixedSize(700, 600)
        
        self.motor = MotorVision()
        self.accesorio = "gafas"

        self.label_camara = QLabel()
        self.label_camara.setFixedSize(640, 480)

        self.btn_gafas = QPushButton("USAR GAFAS")
        self.btn_gafas.clicked.connect(lambda: self.cambiar_item("gafas"))

        self.btn_sombrero = QPushButton("USAR SOMBRERO")
        self.btn_sombrero.clicked.connect(lambda: self.cambiar_item("sombrero"))

        layout = QVBoxLayout()
        layout.addWidget(self.label_camara, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.btn_gafas)
        layout.addWidget(self.btn_sombrero)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def cambiar_item(self, item):
        self.accesorio = item

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            res = self.motor.procesar_frame(frame)
            frame = self.motor.dibujar_accesorio(frame, res, self.accesorio)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.label_camara.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = ProbadorApp()
    main_win.show()
    sys.exit(app.exec())
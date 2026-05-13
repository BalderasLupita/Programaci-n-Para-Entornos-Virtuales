import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QComboBox, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# Alias para la nueva API de Vision
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class ContadorEjercicios(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💪 Contador Pro - MediaPipe Vision API")
        self.setGeometry(100, 100, 1300, 800)
        
        # --- Configuración de MediaPipe Vision ---
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
            running_mode=VisionRunningMode.VIDEO, # Optimizado para streaming
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        
        # Captura de video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        # Variables de estado
        self.ejercicio_actual = "sentadilla"
        self.contador = 0
        self.etapa = "arriba"
        self.umbral_abajo = 90
        self.umbral_arriba = 160
        self.historial_angulos = []
        
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Panel de video
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        layout.addWidget(self.label_video, 3)
        
        # Panel de control
        panel_control = QWidget()
        panel_control.setFixedWidth(300)
        layout_ctrl = QVBoxLayout(panel_control)
        
        # ComboBox
        self.combo = QComboBox()
        self.combo.addItems(["sentadilla", "flexion", "abdominal"])
        self.combo.currentTextChanged.connect(self.cambiar_ejercicio)
        layout_ctrl.addWidget(QLabel("Seleccionar Ejercicio:"))
        layout_ctrl.addWidget(self.combo)
        
        # Stats
        self.label_contador = QLabel("0")
        self.label_contador.setStyleSheet("font-size: 80px; color: #4CAF50; font-weight: bold;")
        self.label_contador.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_ctrl.addWidget(self.label_contador)
        
        self.label_feedback = QLabel("¡Listo!")
        layout_ctrl.addWidget(self.label_feedback)
        
        btn_reset = QPushButton("Reiniciar")
        btn_reset.clicked.connect(self.reiniciar_contador)
        layout_ctrl.addWidget(btn_reset)
        
        layout_ctrl.addStretch()
        layout.addWidget(panel_control)

    def calcular_angulo(self, a, b, c, landmarks, w, h):
        # En la nueva API, landmarks es una lista de objetos con x, y
        p_a = np.array([landmarks[a].x * w, landmarks[a].y * h])
        p_b = np.array([landmarks[b].x * w, landmarks[b].y * h])
        p_c = np.array([landmarks[c].x * w, landmarks[c].y * h])
        
        ba = p_a - p_b
        bc = p_c - p_b
        
        cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angulo = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return angulo, p_a.astype(int), p_b.astype(int), p_c.astype(int)

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        # Convertir a formato MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        
        # Inferencia
        resultados = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        h, w = frame.shape[:2]

        if resultados.pose_landmarks:
            for pose_lms in resultados.pose_landmarks:
                # Lógica por ejercicio
                if self.ejercicio_actual == "sentadilla":
                    angulo, p1, p2, p3 = self.calcular_angulo(23, 25, 27, pose_lms, w, h)
                elif self.ejercicio_actual == "flexion":
                    angulo, p1, p2, p3 = self.calcular_angulo(11, 13, 15, pose_lms, w, h)
                else: # abdominal
                    angulo, p1, p2, p3 = self.calcular_angulo(11, 23, 25, pose_lms, w, h)

                # Dibujar esqueleto básico (Visualización manual simple)
                cv2.line(frame, tuple(p1), tuple(p2), (255, 255, 0), 3)
                cv2.line(frame, tuple(p2), tuple(p3), (255, 255, 0), 3)
                for p in [p1, p2, p3]: cv2.circle(frame, tuple(p), 6, (0, 0, 255), -1)

                self.contar_repeticiones(angulo)
                cv2.putText(frame, f"{angulo:.0f} deg", tuple(p2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        self.mostrar_imagen(frame)

    def contar_repeticiones(self, angulo):
        if angulo < self.umbral_abajo and self.etapa == "arriba":
            self.etapa = "abajo"
            self.label_feedback.setText("⬇️ ¡Baja!")
        elif angulo > self.umbral_arriba and self.etapa == "abajo":
            self.etapa = "arriba"
            self.contador += 1
            self.label_contador.setText(str(self.contador))
            self.label_feedback.setText("✅ ¡Bien!")

    def cambiar_ejercicio(self, ej):
        self.ejercicio_actual = ej
        self.reiniciar_contador()

    def reiniciar_contador(self):
        self.contador = 0
        self.label_contador.setText("0")
        self.etapa = "arriba"

    def mostrar_imagen(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        self.landmarker.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ContadorEjercicios()
    window.show()
    sys.exit(app.exec())
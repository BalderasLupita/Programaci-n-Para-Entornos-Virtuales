import sys
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QSlider, QComboBox, QColorDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class MallaFacialArtistica(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 Malla Facial Artística - MediaPipe Vision")
        self.setGeometry(100, 100, 1300, 800)
        
        # --- Configuración de MediaPipe Vision ---
        # Nota: Debes tener el archivo 'face_landmarker.task' en la misma carpeta
        # Descárgalo aquí: https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Configuración artística
        self.estilo = "contorno"
        self.color = (0, 255, 0)  # BGR para OpenCV
        self.grosor = 1
        self.tam_punto = 2
        self.efecto_arcoiris = False
        
        # Captura de video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        self.expresion_actual = "Neutral"
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Panel de video
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        
        self.expresion_label = QLabel("😐 Expresión: Neutral")
        self.expresion_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout_video.addWidget(self.expresion_label)
        layout.addWidget(panel_video, 3)
        
        # Panel de control
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        # Grupo: Estilo
        grupo_estilo = QGroupBox("🎭 Estilo de dibujo")
        layout_estilo = QVBoxLayout()
        self.combo_estilo = QComboBox()
        self.combo_estilo.addItems(["contorno", "puntos", "solo_ojos", "solo_boca"])
        self.combo_estilo.currentTextChanged.connect(lambda v: setattr(self, 'estilo', v))
        layout_estilo.addWidget(self.combo_estilo)
        grupo_estilo.setLayout(layout_estilo)
        layout_control.addWidget(grupo_estilo)
        
        # Grupo: Color
        grupo_color = QGroupBox("🎨 Color")
        layout_color = QVBoxLayout()
        btn_color = QPushButton("Seleccionar color")
        btn_color.clicked.connect(self.seleccionar_color)
        layout_color.addWidget(btn_color)
        
        self.cb_arcoiris = QPushButton("🌈 Activar arcoíris")
        self.cb_arcoiris.setCheckable(True)
        self.cb_arcoiris.toggled.connect(lambda v: setattr(self, 'efecto_arcoiris', v))
        layout_color.addWidget(self.cb_arcoiris)
        
        grupo_color.setLayout(layout_color)
        layout_control.addWidget(grupo_color)
        
        # Grupo: Información
        grupo_info = QGroupBox("ℹ️ Info")
        layout_info = QVBoxLayout()
        self.info_label = QLabel("Esperando rostro...")
        layout_info.addWidget(self.info_label)
        grupo_info.setLayout(layout_info)
        layout_control.addWidget(grupo_info)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def seleccionar_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = (color.blue(), color.green(), color.red())

    def dibujar_estilo(self, frame, landmarks, ancho, alto):
        """Dibuja basado en la lista de landmarks (NormalizedLandmarks)"""
        if self.estilo == "contorno":
            indices_contorno = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                               397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                               172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            
            puntos = []
            for idx in indices_contorno:
                lm = landmarks[idx]
                puntos.append((int(lm.x * ancho), int(lm.y * alto)))
            
            for i in range(len(puntos)-1):
                cv2.line(frame, puntos[i], puntos[i+1], self.color, self.grosor)
        
        elif self.estilo == "puntos":
            for lm in landmarks:
                x, y = int(lm.x * ancho), int(lm.y * alto)
                color = (x % 255, y % 255, (x+y) % 255) if self.efecto_arcoiris else self.color
                cv2.circle(frame, (x, y), self.tam_punto, color, -1)

        elif self.estilo == "solo_ojos":
            for idx in [33, 133, 159, 145, 362, 263, 386, 374]: # Simplificado
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * ancho), int(lm.y * alto)), 3, self.color, -1)

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convertir frame de OpenCV (BGR) a MP Image (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Inferencia
        resultados = self.detector.detect(mp_image)
        
        if resultados.face_landmarks:
            for face_landmarks in resultados.face_landmarks:
                self.dibujar_estilo(frame, face_landmarks, w, h)
                
                # Ejemplo de detección de sonrisa básica usando blendshapes
                if resultados.face_blendshapes:
                    # El índice 44 suele ser 'mouthSmileLeft' en el modelo de MP
                    sonrisa_izq = resultados.face_blendshapes[0][44].score
                    sonrisa_der = resultados.face_blendshapes[0][45].score
                    
                    if (sonrisa_izq + sonrisa_der) / 2 > 0.4:
                        self.expresion_actual = "😊 Sonrisa"
                    else:
                        self.expresion_actual = "😐 Neutral"
            
            self.info_label.setText(f"Rostros: {len(resultados.face_landmarks)}")
            self.expresion_label.setText(f"🎭 Expresión: {self.expresion_actual}")
        else:
            self.info_label.setText("No se detectó rostro")
        
        self.mostrar_imagen(frame)

    def mostrar_imagen(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label_video.setPixmap(pixmap.scaled(self.label_video.size(), 
                                   Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        self.detector.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = MallaFacialArtistica()
    ventana.show()
    sys.exit(app.exec())
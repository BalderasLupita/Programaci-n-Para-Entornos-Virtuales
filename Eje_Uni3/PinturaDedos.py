import sys
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QColorDialog, QSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class PinturaDedos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖌️ Pintura con Dedos - MediaPipe Vision")
        self.setGeometry(100, 100, 1400, 800)
        
        # --- Configuración de MediaPipe Vision ---
        # Nota: Asegúrate de descargar 'hand_landmarker.task' de la web oficial de MediaPipe
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Captura de video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        # Lienzo para dibujar
        self.lienzo = None
        self.ultima_posicion = None
        
        # Configuración de dibujo
        self.color_actual = (0, 255, 0)
        self.grosor_actual = 5
        self.modo_borrador = False
        
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
        layout.addWidget(panel_video, 3)
        
        # Panel de control
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        # Grupo: Color
        grupo_color = QGroupBox("🎨 Color")
        layout_col = QVBoxLayout()
        btn_color = QPushButton("Seleccionar color")
        btn_color.clicked.connect(self.seleccionar_color)
        btn_borrador = QPushButton("🧼 Activar borrador")
        btn_borrador.setCheckable(True)
        btn_borrador.toggled.connect(lambda v: setattr(self, 'modo_borrador', v))
        layout_col.addWidget(btn_color)
        layout_col.addWidget(btn_borrador)
        grupo_color.setLayout(layout_col)
        layout_control.addWidget(grupo_color)
        
        # Grupo: Grosor
        grupo_grosor = QGroupBox("✏️ Grosor")
        layout_gr = QVBoxLayout()
        self.spin_grosor = QSpinBox()
        self.spin_grosor.setRange(1, 20)
        self.spin_grosor.setValue(5)
        self.spin_grosor.valueChanged.connect(lambda v: setattr(self, 'grosor_actual', v))
        layout_gr.addWidget(self.spin_grosor)
        grupo_grosor.setLayout(layout_gr)
        layout_control.addWidget(grupo_grosor)
        
        # Grupo: Acciones
        grupo_acciones = QGroupBox("💾 Acciones")
        layout_acc = QVBoxLayout()
        btn_limpiar = QPushButton("🧹 Limpiar lienzo")
        btn_limpiar.clicked.connect(self.limpiar_lienzo)
        layout_acc.addWidget(btn_limpiar)
        grupo_acciones.setLayout(layout_acc)
        layout_control.addWidget(grupo_acciones)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
        self.limpiar_lienzo()

    def seleccionar_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_actual = (color.blue(), color.green(), color.red())

    def limpiar_lienzo(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                self.lienzo = np.zeros((h, w, 3), dtype=np.uint8)

    def detectar_gesto(self, landmarks):
        # En la nueva API, landmarks es una lista de objetos con x, y, z
        p_indice = landmarks[8]
        p_medio = landmarks[12]
        p_anular = landmarks[16]
        p_menique = landmarks[20]
        
        n_indice = landmarks[6]
        n_medio = landmarks[10]
        n_anular = landmarks[14]
        n_menique = landmarks[18]
        
        indice_ext = p_indice.y < n_indice.y
        medio_ext = p_medio.y < n_medio.y
        anular_ext = p_anular.y < n_anular.y
        menique_ext = p_menique.y < n_menique.y
        
        if indice_ext and not medio_ext and not anular_ext and not menique_ext:
            return "dibujar"
        elif indice_ext and medio_ext and not anular_ext and not menique_ext:
            return "borrador"
        elif indice_ext and medio_ext and anular_ext and menique_ext:
            return "limpiar"
        return "none"

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1) # Espejo para dibujo natural
        h, w = frame.shape[:2]
        
        if self.lienzo is None or self.lienzo.shape != frame.shape:
            self.lienzo = np.zeros_like(frame)
        
        # Conversión para MediaPipe Vision
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Procesar
        resultados = self.detector.detect(mp_image)
        
        if resultados.hand_landmarks:
            for hand_landmarks in resultados.hand_landmarks:
                # El índice es el landmark 8
                x = int(hand_landmarks[8].x * w)
                y = int(hand_landmarks[8].y * h)
                
                gesto = self.detectar_gesto(hand_landmarks)
                
                # Lógica de dibujo
                if gesto == "dibujar" and not self.modo_borrador:
                    if self.ultima_posicion:
                        cv2.line(self.lienzo, self.ultima_posicion, (x, y), self.color_actual, self.grosor_actual)
                    self.ultima_posicion = (x, y)
                elif gesto == "borrador" or (gesto == "dibujar" and self.modo_borrador):
                    if self.ultima_posicion:
                        cv2.line(self.lienzo, self.ultima_posicion, (x, y), (0,0,0), self.grosor_actual*4)
                    self.ultima_posicion = (x, y)
                elif gesto == "limpiar":
                    self.limpiar_lienzo()
                    self.ultima_posicion = None
                else:
                    self.ultima_posicion = None

                # Dibujar un pequeño puntero en el frame de video
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

        # Combinar
        resultado = cv2.addWeighted(frame, 1.0, self.lienzo, 0.8, 0)
        self.mostrar_en_label(resultado)

    def mostrar_en_label(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label_video.setPixmap(pixmap.scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        self.detector.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = PinturaDedos()
    ventana.show()
    sys.exit(app.exec())
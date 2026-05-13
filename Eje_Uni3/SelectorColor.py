import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QGroupBox, QComboBox)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QImage, QPixmap

class SelectorColorMagico(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 Selector de Color Mágico - Efecto Cine")
        self.setGeometry(100, 100, 1300, 800)
        
        # Variables de captura
        self.cap = None
        self.camara_activa = False
        self.frame_actual = None
        
        # Rango HSV inicial (Todo el espectro)
        self.h_min, self.h_max = 0, 179
        self.s_min, self.s_max = 0, 255
        self.v_min, self.v_max = 0, 255
        
        # Colores predefinidos ajustados para ser más tolerantes
        # Formato: (h_min, h_max, s_min, s_max, v_min, v_max)
        self.colores_preset = {
            "Personalizado": (0, 179, 0, 255, 0, 255),
            "Rojo": (0, 10, 70, 255, 50, 255),
            "Verde": (35, 85, 70, 255, 50, 255),
            "Azul": (100, 130, 70, 255, 50, 255),
            "Amarillo": (20, 35, 70, 255, 50, 255),
            "Naranja": (10, 20, 70, 255, 50, 255),
            "Rosa": (140, 170, 70, 255, 50, 255),
        }
        
        self.setup_ui()
        self.setup_camara()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # --- PANEL DE VIDEO ---
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        layout.addWidget(panel_video, 3)
        
        # --- PANEL DE CONTROL ---
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        # Presets
        grupo_preset = QGroupBox("🎨 Colores Predefinidos")
        layout_p = QVBoxLayout()
        self.combo_colores = QComboBox()
        self.combo_colores.addItems(self.colores_preset.keys())
        self.combo_colores.currentTextChanged.connect(self.cambiar_preset)
        layout_p.addWidget(self.combo_colores)
        grupo_preset.setLayout(layout_p)
        layout_control.addWidget(grupo_preset)
        
        # Sliders HSV
        grupo_hsv = QGroupBox("🎚️ Ajuste Fino (HSV)")
        layout_h = QVBoxLayout()
        self.sliders = {}
        
        controles = [
            ('h_min', 'Hue Mín', 0, 179, 0),
            ('h_max', 'Hue Máx', 0, 179, 179),
            ('s_min', 'Sat Mín', 0, 255, 0),
            ('s_max', 'Sat Máx', 0, 255, 255),
            ('v_min', 'Val Mín', 0, 255, 0),
            ('v_max', 'Val Máx', 0, 255, 255),
        ]
        
        for key, name, mini, maxi, default in controles:
            layout_h.addWidget(QLabel(name))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(mini, maxi)
            slider.setValue(default)
            slider.valueChanged.connect(lambda v, k=key: self.actualizar_hsv(k, v))
            layout_h.addWidget(slider)
            self.sliders[key] = slider
            
        grupo_hsv.setLayout(layout_h)
        layout_control.addWidget(grupo_hsv)
        
        # Botones
        btn_captura = QPushButton("📸 Guardar Instantánea")
        btn_captura.setFixedHeight(40)
        btn_captura.clicked.connect(self.guardar_instantanea)
        layout_control.addWidget(btn_captura)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)

    def setup_camara(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.camara_activa = True
            self.timer = QTimer()
            self.timer.timeout.connect(self.actualizar_frame)
            self.timer.start(30)

    def actualizar_hsv(self, parametro, valor):
        setattr(self, parametro, valor)

    def cambiar_preset(self, nombre):
        if nombre in self.colores_preset:
            val = self.colores_preset[nombre]
            self.h_min, self.h_max, self.s_min, self.s_max, self.v_min, self.v_max = val
            # Actualizar posiciones de sliders
            for i, key in enumerate(['h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max']):
                self.sliders[key].setValue(val[i])

    def aplicar_efecto_cine(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Máscara de color
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Limpieza de ruido
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (7,7), 0)
        
        # Crear versión Gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Combinar
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        resultado = (frame * mask_3ch + gray_bgr * (1 - mask_3ch)).astype(np.uint8)
        return resultado

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Procesar
            self.frame_actual = self.aplicar_efecto_cine(frame)
            
            # Convertir para PyQt
            rgb = cv2.cvtColor(self.frame_actual, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img).scaled(self.label_video.size(), 
                                                     Qt.AspectRatioMode.KeepAspectRatio, 
                                                     Qt.TransformationMode.SmoothTransformation)
            self.label_video.setPixmap(pixmap)

    def guardar_instantanea(self):
        if self.frame_actual is not None:
            nombre = f"captura_{QDateTime.currentDateTime().toString('hhmmss')}.png"
            cv2.imwrite(nombre, self.frame_actual)
            print(f"Guardado como {nombre}")

    def closeEvent(self, event):
        if self.cap: self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = SelectorColorMagico()
    ventana.show()
    sys.exit(app.exec())
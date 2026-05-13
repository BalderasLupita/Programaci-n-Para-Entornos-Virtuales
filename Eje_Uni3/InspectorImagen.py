
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QFileDialog, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class InspectorImagen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Inspector de Imagen - Capítulo 1")
        self.setGeometry(100, 100, 1200, 700)
        
        # Variables de imagen
        self.imagen_original = None
        self.imagen_procesada = None
        
        # Valores de ajuste
        self.brillo = 0
        self.contraste = 1.0
        self.saturacion = 1.0
        
        # Configurar UI
        self.setup_ui()
        
        # Timer para actualización automática
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_imagen)
        self.timer.start(50)  # 20 fps
        
    def setup_ui(self):
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        
        # Layout principal horizontal
        layout = QHBoxLayout(central)
        
        # Panel izquierdo: imagen
        self.label_imagen = QLabel()
        self.label_imagen.setMinimumSize(800, 600)
        self.label_imagen.setStyleSheet("border: 2px solid #333; background-color: #222;")
        self.label_imagen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label_imagen, 3)
        
        # Panel derecho: controles
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        # Grupo: Carga de imagen
        grupo_carga = QGroupBox("📁 Cargar Imagen")
        layout_carga = QVBoxLayout()
        
        btn_cargar = QPushButton("Seleccionar imagen...")
        btn_cargar.clicked.connect(self.cargar_imagen)
        layout_carga.addWidget(btn_cargar)
        
        self.info_label = QLabel("No hay imagen cargada")
        layout_carga.addWidget(self.info_label)
        
        grupo_carga.setLayout(layout_carga)
        layout_control.addWidget(grupo_carga)
        
        # Grupo: Ajustes básicos
        grupo_ajustes = QGroupBox("🎨 Ajustes")
        layout_ajustes = QVBoxLayout()
        
        # Brillo
        layout_ajustes.addWidget(QLabel("Brillo:"))
        self.slider_brillo = QSlider(Qt.Orientation.Horizontal)
        self.slider_brillo.setRange(-100, 100)
        self.slider_brillo.setValue(0)
        self.slider_brillo.valueChanged.connect(self.actualizar_brillo)
        layout_ajustes.addWidget(self.slider_brillo)
        
        # Contraste
        layout_ajustes.addWidget(QLabel("Contraste:"))
        self.slider_contraste = QSlider(Qt.Orientation.Horizontal)
        self.slider_contraste.setRange(0, 200)
        self.slider_contraste.setValue(100)
        self.slider_contraste.valueChanged.connect(self.actualizar_contraste)
        layout_ajustes.addWidget(self.slider_contraste)
        
        # Saturación
        layout_ajustes.addWidget(QLabel("Saturación:"))
        self.slider_saturacion = QSlider(Qt.Orientation.Horizontal)
        self.slider_saturacion.setRange(0, 200)
        self.slider_saturacion.setValue(100)
        self.slider_saturacion.valueChanged.connect(self.actualizar_saturacion)
        layout_ajustes.addWidget(self.slider_saturacion)
        
        grupo_ajustes.setLayout(layout_ajustes)
        layout_control.addWidget(grupo_ajustes)
        
        # Grupo: Información técnica
        grupo_info = QGroupBox("📊 Info Técnica")
        layout_info = QVBoxLayout()
        
        self.tec_label = QLabel("Dimensiones: -\nCanales: -\nTamaño: -")
        layout_info.addWidget(self.tec_label)
        
        grupo_info.setLayout(layout_info)
        layout_control.addWidget(grupo_info)
        
        # Espaciador
        layout_control.addStretch()
        
        layout.addWidget(panel_control, 1)
        
    def cargar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "", 
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if archivo:
            self.imagen_original = cv2.imread(archivo)
            if self.imagen_original is not None:
                self.imagen_procesada = self.imagen_original.copy()
                self.info_label.setText(f"✅ Cargada: {archivo.split('/')[-1]}")
                self.actualizar_info_tecnica()
                self.mostrar_imagen()
    
    def actualizar_brillo(self, valor):
        self.brillo = valor
        self.procesar_imagen()
    
    def actualizar_contraste(self, valor):
        self.contraste = valor / 100.0
        self.procesar_imagen()
    
    def actualizar_saturacion(self, valor):
        self.saturacion = valor / 100.0
        self.procesar_imagen()
    
    def procesar_imagen(self):
        if self.imagen_original is None:
            return
            
        # Aplicar brillo y contraste
        img = cv2.convertScaleAbs(
            self.imagen_original, 
            alpha=self.contraste, 
            beta=self.brillo
        )
        
        # Aplicar saturación (convertir a HSV y ajustar)
        if self.saturacion != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * self.saturacion
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        self.imagen_procesada = img
        self.mostrar_imagen()
    
    def mostrar_imagen(self):
        if self.imagen_procesada is None:
            return
            
        # Convertir BGR a RGB para Qt
        rgb = cv2.cvtColor(self.imagen_procesada, cv2.COLOR_BGR2RGB)
        
        # Crear QImage
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Escalar para mantener proporción
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(
            self.label_imagen.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.label_imagen.setPixmap(pixmap)
    
    def actualizar_info_tecnica(self):
        if self.imagen_original is not None:
            h, w = self.imagen_original.shape[:2]
            canales = self.imagen_original.shape[2] if len(self.imagen_original.shape) == 3 else 1
            tamaño = self.imagen_original.size * self.imagen_original.itemsize / 1024
            
            self.tec_label.setText(
                f"Dimensiones: {w} x {h} px\n"
                f"Canales: {canales}\n"
                f"Tamaño en memoria: {tamaño:.1f} KB\n"
                f"Formato: BGR (OpenCV)"
            )
    
    def actualizar_imagen(self):
        """Se llama periódicamente para actualizar la vista"""
        if self.imagen_procesada is not None:
            # Aquí podrías añadir efectos dinámicos
            pass

def main():
    app = QApplication(sys.argv)
    ventana = InspectorImagen()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()



import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QSlider, QCheckBox, QTableWidget, QTableWidgetItem,
                             QHeaderView)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QImage, QPixmap, QFont

class DetectorFiguras(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Detector de Figuras Geométricas - Capítulo 3")
        self.setGeometry(100, 100, 1400, 800)
        
        # Variables
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        # Parámetros ajustables
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.min_area = 500
        self.detectar_circulos = True
        self.detectar_poligonos = True
        self.mostrar_bordes = False
        
        # Contadores
        self.contadores = {
            "Triángulo": 0,
            "Cuadrado": 0,
            "Rectángulo": 0,
            "Círculo": 0,
            "Pentágono": 0,
            "Hexágono": 0,
            "Desconocido": 0
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        # Layout principal
        layout = QHBoxLayout(central)
        
        # Panel de video
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        
        self.label_video = QLabel()
        self.label_video.setMinimumSize(900, 600)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        
        # Checkbox para bordes
        self.cb_bordes = QCheckBox("Mostrar bordes Canny")
        self.cb_bordes.stateChanged.connect(lambda v: setattr(self, 'mostrar_bordes', v))
        layout_video.addWidget(self.cb_bordes)
        
        layout.addWidget(panel_video, 3)
        
        # Panel de control
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        # Grupo: Parámetros Canny
        grupo_canny = QGroupBox(" Parámetros Canny")
        layout_canny = QVBoxLayout()
        
        layout_canny.addWidget(QLabel("Umbral 1:"))
        slider_c1 = QSlider(Qt.Orientation.Horizontal)
        slider_c1.setRange(0, 255)
        slider_c1.setValue(50)
        slider_c1.valueChanged.connect(lambda v: setattr(self, 'canny_threshold1', v))
        layout_canny.addWidget(slider_c1)
        
        layout_canny.addWidget(QLabel("Umbral 2:"))
        slider_c2 = QSlider(Qt.Orientation.Horizontal)
        slider_c2.setRange(0, 255)
        slider_c2.setValue(150)
        slider_c2.valueChanged.connect(lambda v: setattr(self, 'canny_threshold2', v))
        layout_canny.addWidget(slider_c2)
        
        grupo_canny.setLayout(layout_canny)
        layout_control.addWidget(grupo_canny)
        
        # Grupo: Filtros
        grupo_filtros = QGroupBox(" Tipos a detectar")
        layout_filtros = QVBoxLayout()
        
        self.cb_circulos = QCheckBox("Círculos (Hough)")
        self.cb_circulos.setChecked(True)
        self.cb_circulos.stateChanged.connect(lambda v: setattr(self, 'detectar_circulos', v))
        layout_filtros.addWidget(self.cb_circulos)
        
        self.cb_poligonos = QCheckBox("Polígonos")
        self.cb_poligonos.setChecked(True)
        self.cb_poligonos.stateChanged.connect(lambda v: setattr(self, 'detectar_poligonos', v))
        layout_filtros.addWidget(self.cb_poligonos)
        
        grupo_filtros.setLayout(layout_filtros)
        layout_control.addWidget(grupo_filtros)
        
        # Grupo: Área mínima
        grupo_area = QGroupBox(" Área mínima")
        layout_area = QVBoxLayout()
        
        slider_area = QSlider(Qt.Orientation.Horizontal)
        slider_area.setRange(100, 5000)
        slider_area.setValue(500)
        slider_area.valueChanged.connect(lambda v: setattr(self, 'min_area', v))
        layout_area.addWidget(slider_area)
        self.label_area = QLabel("500 px²")
        slider_area.valueChanged.connect(lambda v: self.label_area.setText(f"{v} px²"))
        layout_area.addWidget(self.label_area)
        
        grupo_area.setLayout(layout_area)
        layout_control.addWidget(grupo_area)
        
        # Grupo: Estadísticas
        grupo_stats = QGroupBox(" Conteo de formas")
        layout_stats = QVBoxLayout()
        
        self.tabla_stats = QTableWidget(7, 2)
        self.tabla_stats.setHorizontalHeaderLabels(["Forma", "Cantidad"])
        self.tabla_stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Llenar tabla
        formas = list(self.contadores.keys())
        for i, forma in enumerate(formas):
            self.tabla_stats.setItem(i, 0, QTableWidgetItem(forma))
            self.tabla_stats.setItem(i, 1, QTableWidgetItem("0"))
        
        layout_stats.addWidget(self.tabla_stats)
        
        btn_reset = QPushButton(" Reiniciar contadores")
        btn_reset.clicked.connect(self.reiniciar_contadores)
        layout_stats.addWidget(btn_reset)
        
        grupo_stats.setLayout(layout_stats)
        layout_control.addWidget(grupo_stats)
        
        # Botón captura
        btn_captura = QPushButton(" Capturar con detección")
        btn_captura.clicked.connect(self.guardar_captura)
        layout_control.addWidget(btn_captura)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
        
    def detectar_forma(self, contorno):
        """Identifica la forma de un contorno"""
        peri = cv2.arcLength(contorno, True)
        aproximacion = cv2.approxPolyDP(contorno, 0.04 * peri, True)
        vertices = len(aproximacion)
        
        # Polígonos regulares
        if vertices == 3:
            return "Triángulo"
        elif vertices == 4:
            # Verificar si es cuadrado o rectángulo
            x, y, w, h = cv2.boundingRect(aproximacion)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                return "Cuadrado"
            else:
                return "Rectángulo"
        elif vertices == 5:
            return "Pentágono"
        elif vertices == 6:
            return "Hexágono"
        elif vertices > 6:
            # Posible círculo
            area = cv2.contourArea(contorno)
            (x, y), radio = cv2.minEnclosingCircle(contorno)
            area_circulo = np.pi * radio ** 2
            
            if abs(area - area_circulo) / area_circulo < 0.2:
                return "Círculo"
        
        return "Desconocido"
    
    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Reiniciar contadores para este frame
        contadores_frame = {k: 0 for k in self.contadores.keys()}
        
        # Preprocesamiento
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
        
        # Bordes Canny
        bordes = cv2.Canny(desenfoque, self.canny_threshold1, self.canny_threshold2)
        
        if self.detectar_poligonos:
            # Encontrar contornos
            contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area < self.min_area:
                    continue
                
                forma = self.detectar_forma(contorno)
                contadores_frame[forma] += 1
                
                # Dibujar
                color = self.color_para_forma(forma)
                cv2.drawContours(frame, [contorno], -1, color, 2)
                
                # Etiqueta
                M = cv2.moments(contorno)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, forma, (cX - 30, cY - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.detectar_circulos:
            # Detección específica de círculos
            circulos = cv2.HoughCircles(desenfoque, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=50, param2=30, minRadius=10, maxRadius=100)
            
            if circulos is not None:
                circulos = np.round(circulos[0, :]).astype("int")
                for (x, y, r) in circulos:
                    # Verificar que no esté ya detectado
                    cv2.circle(frame, (x, y), r, (255, 0, 255), 2)
                    cv2.putText(frame, "Círculo", (x - 30, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    contadores_frame["Círculo"] += 1
        
        # Actualizar tabla
        for i, (forma, conteo) in enumerate(contadores_frame.items()):
            self.tabla_stats.setItem(i, 1, QTableWidgetItem(str(conteo)))
        
        # Mostrar
        if self.mostrar_bordes:
            # Convertir bordes a BGR para mostrar
            bordes_bgr = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
            self.mostrar_imagen(bordes_bgr)
        else:
            self.mostrar_imagen(frame)
    
    def color_para_forma(self, forma):
        """Devuelve un color BGR para cada forma"""
        colores = {
            "Triángulo": (0, 255, 0),    # Verde
            "Cuadrado": (255, 0, 0),      # Azul
            "Rectángulo": (255, 255, 0),  # Cyan
            "Círculo": (0, 0, 255),       # Rojo
            "Pentágono": (255, 0, 255),   # Magenta
            "Hexágono": (0, 255, 255),    # Amarillo
            "Desconocido": (128, 128, 128) # Gris
        }
        return colores.get(forma, (255, 255, 255))
    
    def mostrar_imagen(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(
            self.label_video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label_video.setPixmap(pixmap)
    
    def reiniciar_contadores(self):
        for i in range(self.tabla_stats.rowCount()):
            self.tabla_stats.setItem(i, 1, QTableWidgetItem("0"))
    
    def guardar_captura(self):
        ret, frame = self.cap.read()
        if ret:
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            cv2.imwrite(f"deteccion_formas_{timestamp}.png", frame)
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ventana = DetectorFiguras()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


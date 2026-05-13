
import sys
import cv2
import cv2.aruco as aruco
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QLineEdit, QTextEdit, QColorDialog, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

class TarjetaPresentacionAR(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📇 Tarjeta de Presentación AR - Capítulo 9")
        self.setGeometry(100, 100, 1400, 800)
        
        # Configurar detector ArUco
        self.diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parametros = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.diccionario, self.parametros)
        
        # Parámetros de cámara (aproximados)
        self.matriz_camara = np.array([[1000, 0, 640],
                                       [0, 1000, 360],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coefs = np.zeros((4, 1))
        self.tamanio_marcador = 0.05  # 5 cm
        
        # Información de la tarjeta
        self.info_tarjetas = {
            0: {
                "nombre": "Ana García",
                "cargo": "Ingeniera AR",
                "empresa": "TechVision",
                "email": "ana@techvision.com",
                "telefono": "+34 123 456 789",
                "color": (255, 0, 0)  # Azul
            },
            1: {
                "nombre": "Carlos López",
                "cargo": "Desarrollador Senior",
                "empresa": "AR Solutions",
                "email": "carlos@arsolutions.com",
                "telefono": "+34 987 654 321",
                "color": (0, 255, 0)  # Verde
            }
        }
        
        # Captura de video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
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
        
        # Panel de edición
        panel_edicion = QWidget()
        panel_edicion.setMaximumWidth(400)
        layout_edicion = QVBoxLayout(panel_edicion)
        
        # Grupo: Selección de marcador
        grupo_seleccion = QGroupBox("🎯 Seleccionar Marcador")
        layout_seleccion = QVBoxLayout()
        
        self.label_id = QLabel("ID del marcador: 0")
        layout_seleccion.addWidget(self.label_id)
        
        btn_id_menos = QPushButton("◀ ID anterior")
        btn_id_menos.clicked.connect(lambda: self.cambiar_id(-1))
        layout_seleccion.addWidget(btn_id_menos)
        
        btn_id_mas = QPushButton("ID siguiente ▶")
        btn_id_mas.clicked.connect(lambda: self.cambiar_id(1))
        layout_seleccion.addWidget(btn_id_mas)
        
        grupo_seleccion.setLayout(layout_seleccion)
        layout_edicion.addWidget(grupo_seleccion)
        
        # Grupo: Editar información
        grupo_edicion = QGroupBox("✏️ Editar Información")
        layout_edicion_info = QVBoxLayout()
        
        layout_edicion_info.addWidget(QLabel("Nombre:"))
        self.input_nombre = QLineEdit()
        self.input_nombre.textChanged.connect(self.guardar_cambios)
        layout_edicion_info.addWidget(self.input_nombre)
        
        layout_edicion_info.addWidget(QLabel("Cargo:"))
        self.input_cargo = QLineEdit()
        self.input_cargo.textChanged.connect(self.guardar_cambios)
        layout_edicion_info.addWidget(self.input_cargo)
        
        layout_edicion_info.addWidget(QLabel("Empresa:"))
        self.input_empresa = QLineEdit()
        self.input_empresa.textChanged.connect(self.guardar_cambios)
        layout_edicion_info.addWidget(self.input_empresa)
        
        layout_edicion_info.addWidget(QLabel("Email:"))
        self.input_email = QLineEdit()
        self.input_email.textChanged.connect(self.guardar_cambios)
        layout_edicion_info.addWidget(self.input_email)
        
        layout_edicion_info.addWidget(QLabel("Teléfono:"))
        self.input_telefono = QLineEdit()
        self.input_telefono.textChanged.connect(self.guardar_cambios)
        layout_edicion_info.addWidget(self.input_telefono)
        
        btn_color = QPushButton("🎨 Seleccionar color")
        btn_color.clicked.connect(self.seleccionar_color)
        layout_edicion_info.addWidget(btn_color)
        
        grupo_edicion.setLayout(layout_edicion_info)
        layout_edicion.addWidget(grupo_edicion)
        
        # Grupo: Acciones
        grupo_acciones = QGroupBox("💾 Acciones")
        layout_acciones = QVBoxLayout()
        
        btn_guardar = QPushButton("💾 Guardar tarjeta")
        btn_guardar.clicked.connect(self.guardar_tarjeta)
        layout_acciones.addWidget(btn_guardar)
        
        btn_cargar = QPushButton("📂 Cargar tarjeta")
        btn_cargar.clicked.connect(self.cargar_tarjeta)
        layout_acciones.addWidget(btn_cargar)
        
        grupo_acciones.setLayout(layout_acciones)
        layout_edicion.addWidget(grupo_acciones)
        
        layout_edicion.addStretch()
        layout.addWidget(panel_edicion, 1)
        
        # Cargar datos del ID actual
        self.cargar_info_actual()
    
    def cambiar_id(self, delta):
        """Cambia el ID actual y carga su información"""
        id_actual = int(self.label_id.text().split(":")[1].strip())
        nuevo_id = id_actual + delta
        if 0 <= nuevo_id <= 9:  # IDs del 0 al 9
            self.label_id.setText(f"ID del marcador: {nuevo_id}")
            self.cargar_info_actual()
    
    def cargar_info_actual(self):
        """Carga la información del ID actual en los campos de edición"""
        id_actual = int(self.label_id.text().split(":")[1].strip())
        
        if id_actual in self.info_tarjetas:
            info = self.info_tarjetas[id_actual]
            self.input_nombre.setText(info.get("nombre", ""))
            self.input_cargo.setText(info.get("cargo", ""))
            self.input_empresa.setText(info.get("empresa", ""))
            self.input_email.setText(info.get("email", ""))
            self.input_telefono.setText(info.get("telefono", ""))
        else:
            # Crear entrada nueva
            self.input_nombre.clear()
            self.input_cargo.clear()
            self.input_empresa.clear()
            self.input_email.clear()
            self.input_telefono.clear()
    
    def guardar_cambios(self):
        """Guarda los cambios en el diccionario"""
        id_actual = int(self.label_id.text().split(":")[1].strip())
        
        self.info_tarjetas[id_actual] = {
            "nombre": self.input_nombre.text(),
            "cargo": self.input_cargo.text(),
            "empresa": self.input_empresa.text(),
            "email": self.input_email.text(),
            "telefono": self.input_telefono.text(),
            "color": self.info_tarjetas.get(id_actual, {}).get("color", (0, 255, 0))
        }
    
    def seleccionar_color(self):
        """Abre diálogo para seleccionar color"""
        color = QColorDialog.getColor()
        if color.isValid():
            id_actual = int(self.label_id.text().split(":")[1].strip())
            if id_actual in self.info_tarjetas:
                self.info_tarjetas[id_actual]["color"] = (
                    color.blue(), color.green(), color.red())
    
    def guardar_tarjeta(self):
        """Guarda todas las tarjetas en un archivo"""
        import json
        
        # Convertir colores a lista para JSON
        info_serializable = {}
        for id_tarjeta, info in self.info_tarjetas.items():
            info_serializable[str(id_tarjeta)] = info.copy()
            if "color" in info_serializable[str(id_tarjeta)]:
                info_serializable[str(id_tarjeta)]["color"] = list(info["color"])
        
        archivo, _ = QFileDialog.getSaveFileName(
            self, "Guardar tarjetas", "", "JSON Files (*.json)")
        
        if archivo:
            with open(archivo, 'w') as f:
                json.dump(info_serializable, f, indent=2)
    
    def cargar_tarjeta(self):
        """Carga tarjetas desde un archivo"""
        import json
        
        archivo, _ = QFileDialog.getOpenFileName(
            self, "Cargar tarjetas", "", "JSON Files (*.json)")
        
        if archivo:
            with open(archivo, 'r') as f:
                info_serializable = json.load(f)
            
            # Convertir colores de vuelta a tupla
            self.info_tarjetas = {}
            for id_str, info in info_serializable.items():
                id_int = int(id_str)
                self.info_tarjetas[id_int] = info
                if "color" in info:
                    self.info_tarjetas[id_int]["color"] = tuple(info["color"])
            
            self.cargar_info_actual()
    
    def dibujar_tarjeta(self, frame, info, esquinas):
        """Dibuja una tarjeta de presentación flotante"""
        if not info:
            return
        
        # Obtener posición del marcador
        pts = esquinas[0].astype(int)
        
        # Calcular esquinas para la tarjeta
        x_min = min(pts[:, 0])
        x_max = max(pts[:, 0])
        y_min = min(pts[:, 1])
        y_max = max(pts[:, 1])
        
        # Expandir área para la tarjeta
        expand_x = (x_max - x_min) // 2
        expand_y = (y_max - y_min) // 2
        
        x1 = max(0, x_min - expand_x)
        x2 = min(frame.shape[1], x_max + expand_x)
        y1 = max(0, y_min - expand_y)
        y2 = min(frame.shape[0], y_max + expand_y * 2)
        
        # Dibujar fondo semi-transparente
        overlay = frame.copy()
        color = info.get("color", (0, 255, 0))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Dibujar borde
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Escribir información
        y_pos = y1 + 30
        cv2.putText(frame, f"📇 {info.get('nombre', 'Sin nombre')}", 
                   (x1 + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(frame, f"👔 {info.get('cargo', '')}", 
                   (x1 + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
        cv2.putText(frame, f"🏢 {info.get('empresa', '')}", 
                   (x1 + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
        cv2.putText(frame, f"📧 {info.get('email', '')}", 
                   (x1 + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
        cv2.putText(frame, f"📞 {info.get('telefono', '')}", 
                   (x1 + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Detectar marcadores
        esquinas, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None:
            # Dibujar marcadores detectados
            aruco.drawDetectedMarkers(frame, esquinas, ids)
            
            # Para cada marcador detectado
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.info_tarjetas:
                    # Dibujar tarjeta de presentación
                    self.dibujar_tarjeta(frame, self.info_tarjetas[marker_id], 
                                        esquinas[i])
        
        self.mostrar_imagen(frame)
    
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
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ventana = TarjetaPresentacionAR()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()



import sys
import cv2
import cv2.aruco as aruco
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QListWidget, QFileDialog, QSlider, QCheckBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

class MarcoFotoAR(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖼️ Marco de Foto AR - Capítulo 10")
        self.setGeometry(100, 100, 1400, 800)
        
        # Configurar detector ArUco
        self.diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parametros = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.diccionario, self.parametros)
        
        # Catálogo de imágenes decorativas
        self.catalogo_imagenes = {}
        self.cargar_catalogo_base()
        
        # Imagen seleccionada actualmente
        self.imagen_actual = None
        self.id_imagen_actual = None
        
        # Captura de video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        # Configuración visual
        self.escala_imagen = 1.0
        self.brillo_imagen = 0
        self.borde_decorativo = True
        
        self.setup_ui()
    
    def cargar_catalogo_base(self):
        """Carga imágenes decorativas básicas"""
        # Crear algunas imágenes de prueba
        imagenes_base = {
            "marco_flores": self.crear_marco_flores(),
            "marco_geometrico": self.crear_marco_geometrico(),
            "marco_vintage": self.crear_marco_vintage(),
            "marco_neon": self.crear_marco_neon()
        }
        
        for nombre, img in imagenes_base.items():
            self.catalogo_imagenes[nombre] = img
    
    def crear_marco_flores(self):
        """Crea un marco decorativo con flores"""
        img = np.zeros((400, 400, 4), dtype=np.uint8)
        # Fondo transparente
        img[:,:,3] = 0
        
        # Dibujar flores en las esquinas
        colores = [(255, 192, 203, 255), (255, 255, 0, 255), 
                   (173, 216, 230, 255), (255, 182, 193, 255)]
        
        esquinas = [(50,50), (350,50), (350,350), (50,350)]
        for i, (x,y) in enumerate(esquinas):
            cv2.circle(img, (x,y), 40, colores[i], -1)
            for p in range(5):
                angulo = p * 72 + i * 45
                xp = x + int(30 * np.cos(np.radians(angulo)))
                yp = y + int(30 * np.sin(np.radians(angulo)))
                cv2.circle(img, (xp,yp), 15, colores[i], -1)
        
        # Borde decorativo
        cv2.rectangle(img, (20,20), (380,380), (255,255,255,200), 3)
        
        return img
    
    def crear_marco_geometrico(self):
        """Crea un marco con patrones geométricos"""
        img = np.zeros((400, 400, 4), dtype=np.uint8)
        img[:,:,3] = 0
        
        # Patrón de triángulos
        for i in range(0, 400, 40):
            for j in range(0, 400, 40):
                pts = np.array([[i, j], [i+40, j], [i+20, j+20]], np.int32)
                pts = pts.reshape((-1,1,2))
                color = (0, 255, 0, 100) if (i+j) % 80 == 0 else (255, 0, 0, 100)
                cv2.fillPoly(img, [pts], color)
        
        # Borde
        cv2.rectangle(img, (10,10), (390,390), (255,255,255,255), 5)
        
        return img
    
    def crear_marco_vintage(self):
        """Crea un marco de estilo vintage"""
        img = np.zeros((400, 400, 4), dtype=np.uint8)
        
        # Fondo sepia semi-transparente
        img[:,:,0] = 100  # B
        img[:,:,1] = 150  # G
        img[:,:,2] = 200  # R
        img[:,:,3] = 100  # Alpha
        
        # Patrón de esquinas
        for x, y in [(50,50), (350,50), (350,350), (50,350)]:
            cv2.ellipse(img, (x,y), (50,30), 0, 0, 360, (255,215,0,255), 5)
        
        return img
    
    def crear_marco_neon(self):
        """Crea un marco con efecto neón"""
        img = np.zeros((400, 400, 4), dtype=np.uint8)
        img[:,:,3] = 0
        
        # Líneas neón
        colores = [(0,255,255,255), (255,0,255,255), (255,255,0,255)]
        for i, color in enumerate(colores):
            offset = i * 5
            cv2.rectangle(img, (50+offset,50+offset), 
                         (350-offset,350-offset), color, 2)
        
        return img
    
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
        
        # Grupo: Catálogo de marcos
        grupo_catalogo = QGroupBox("🖼️ Catálogo de Marcos")
        layout_catalogo = QVBoxLayout()
        
        self.lista_marcos = QListWidget()
        for nombre in self.catalogo_imagenes.keys():
            self.lista_marcos.addItem(nombre)
        self.lista_marcos.currentTextChanged.connect(self.seleccionar_marco)
        layout_catalogo.addWidget(self.lista_marcos)
        
        btn_cargar = QPushButton("📂 Cargar imagen propia")
        btn_cargar.clicked.connect(self.cargar_imagen_propia)
        layout_catalogo.addWidget(btn_cargar)
        
        grupo_catalogo.setLayout(layout_catalogo)
        layout_control.addWidget(grupo_catalogo)
        
        # Grupo: Ajustes
        grupo_ajustes = QGroupBox("⚙️ Ajustes")
        layout_ajustes = QVBoxLayout()
        
        layout_ajustes.addWidget(QLabel("Escala:"))
        slider_escala = QSlider(Qt.Orientation.Horizontal)
        slider_escala.setRange(50, 200)
        slider_escala.setValue(100)
        slider_escala.valueChanged.connect(
            lambda v: setattr(self, 'escala_imagen', v/100.0))
        layout_ajustes.addWidget(slider_escala)
        
        layout_ajustes.addWidget(QLabel("Brillo:"))
        slider_brillo = QSlider(Qt.Orientation.Horizontal)
        slider_brillo.setRange(-50, 50)
        slider_brillo.setValue(0)
        slider_brillo.valueChanged.connect(
            lambda v: setattr(self, 'brillo_imagen', v))
        layout_ajustes.addWidget(slider_brillo)
        
        self.cb_borde = QCheckBox("Borde decorativo")
        self.cb_borde.setChecked(True)
        self.cb_borde.stateChanged.connect(
            lambda v: setattr(self, 'borde_decorativo', v))
        layout_ajustes.addWidget(self.cb_borde)
        
        grupo_ajustes.setLayout(layout_ajustes)
        layout_control.addWidget(grupo_ajustes)
        
        # Grupo: Información
        grupo_info = QGroupBox("ℹ️ Información")
        layout_info = QVBoxLayout()
        
        self.info_label = QLabel(
            "Coloca un marcador ArUco\n"
            "para ver el marco seleccionado\n\n"
            "IDs disponibles: 0-9"
        )
        layout_info.addWidget(self.info_label)
        
        grupo_info.setLayout(layout_info)
        layout_control.addWidget(grupo_info)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
    
    def seleccionar_marco(self, nombre):
        """Selecciona un marco del catálogo"""
        if nombre in self.catalogo_imagenes:
            self.imagen_actual = self.catalogo_imagenes[nombre].copy()
            self.id_imagen_actual = nombre
            self.info_label.setText(f"Marco seleccionado: {nombre}")
    
    def cargar_imagen_propia(self):
        """Carga una imagen PNG personalizada"""
        archivo, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen PNG", "", "PNG Images (*.png)")
        
        if archivo:
            img = cv2.imread(archivo, cv2.IMREAD_UNCHANGED)
            if img is not None:
                nombre = archivo.split('/')[-1]
                self.catalogo_imagenes[nombre] = img
                self.lista_marcos.addItem(nombre)
                self.seleccionar_marco(nombre)
    
    def aplicar_ajustes(self, img):
        """Aplica escala y brillo a la imagen"""
        if self.escala_imagen != 1.0:
            h, w = img.shape[:2]
            nuevo_h = int(h * self.escala_imagen)
            nuevo_w = int(w * self.escala_imagen)
            img = cv2.resize(img, (nuevo_w, nuevo_h))
        
        if self.brillo_imagen != 0:
            img[:,:,:3] = cv2.add(img[:,:,:3], self.brillo_imagen)
        
        return img
    
    def superponer_imagen(self, frame, img_superponer, esquinas):
        """Superpone la imagen sobre las esquinas del marcador"""
        h_img, w_img = img_superponer.shape[:2]
        
        # Puntos origen
        pts_origen = np.array([[0, 0], [w_img-1, 0], 
                              [w_img-1, h_img-1], [0, h_img-1]], dtype=np.float32)
        
        # Calcular homografía
        H, _ = cv2.findHomography(pts_origen, esquinas.astype(np.float32))
        
        # Aplicar transformación
        img_warped = cv2.warpPerspective(img_superponer, H, 
                                          (frame.shape[1], frame.shape[0]))
        
        # Crear máscara desde alpha
        if img_superponer.shape[2] == 4:
            mascara = img_warped[:,:,3] / 255.0
        else:
            mascara = np.ones((frame.shape[0], frame.shape[1]), dtype=np.float32)
            cv2.fillConvexPoly(mascara.astype(np.uint8), 
                              esquinas.astype(int), 1)
        
        # Aplicar blending
        for c in range(3):
            frame[:,:,c] = frame[:,:,c] * (1 - mascara) + img_warped[:,:,c] * mascara
        
        return frame
    
    def dibujar_borde_decorativo(self, frame, esquinas):
        """Dibuja un borde decorativo alrededor del marcador"""
        pts = esquinas.astype(int)
        
        # Dibujar líneas punteadas
        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i+1)%4])
            
            # Línea punteada
            dist = np.linalg.norm(np.array(p2) - np.array(p1))
            num_puntos = int(dist / 10)
            
            for j in range(0, num_puntos, 2):
                t1 = j / num_puntos
                t2 = (j+1) / num_puntos
                
                x1 = int(p1[0] + t1 * (p2[0] - p1[0]))
                y1 = int(p1[1] + t1 * (p2[1] - p1[1]))
                x2 = int(p1[0] + t2 * (p2[0] - p1[0]))
                y2 = int(p1[1] + t2 * (p2[1] - p1[1]))
                
                cv2.line(frame, (x1, y1), (x2, y2), (255, 215, 0), 2)
    
    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Detectar marcadores
        esquinas, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None and self.imagen_actual is not None:
            # Dibujar marcadores
            aruco.drawDetectedMarkers(frame, esquinas, ids)
            
            # Aplicar marco sobre el primer marcador
            img_ajustada = self.aplicar_ajustes(self.imagen_actual)
            frame = self.superponer_imagen(frame, img_ajustada, esquinas[0][0])
            
            # Dibujar borde decorativo si está activado
            if self.borde_decorativo:
                self.dibujar_borde_decorativo(frame, esquinas[0][0])
        
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
    ventana = MarcoFotoAR()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


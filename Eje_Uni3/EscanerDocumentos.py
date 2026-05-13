import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QSlider, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap, QAction
from datetime import datetime

class EscanerDocumentos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Escáner de Documentos - Capítulo 4")
        self.setGeometry(100, 100, 1400, 800)
        
        # --- CORRECCIÓN: Indentación de variables ---
        self.imagen_original = None
        self.imagen_procesada = None
        self.puntos_manuales = [] # Lista para almacenar los 4 clics
        
        # Modo: automático o manual
        self.modo_manual = False
        
        self.setup_ui()
        self.setup_menu()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Panel izquierdo
        panel_imagenes = QWidget()
        layout_imagenes = QVBoxLayout(panel_imagenes)
        
        layout_imagenes.addWidget(QLabel(" Original (Haz clic aquí para modo manual):"))
        self.label_original = QLabel()
        self.label_original.setMinimumSize(600, 500)
        self.label_original.setStyleSheet("border: 2px solid #555; background-color: #111;")
        self.label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Habilitar captura de ratón en el label
        self.label_original.setMouseTracking(True)
        layout_imagenes.addWidget(self.label_original)
        
        layout_imagenes.addWidget(QLabel(" Enderezada:"))
        self.label_procesada = QLabel()
        self.label_procesada.setMinimumSize(600, 400)
        self.label_procesada.setStyleSheet("border: 1px solid #333; background-color: #111;")
        self.label_procesada.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_imagenes.addWidget(self.label_procesada)
        
        layout.addWidget(panel_imagenes, 3)
        
        # Panel derecho
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        # Grupos de control
        grupo_carga = QGroupBox(" Carga")
        l_carga = QVBoxLayout(); btn_c = QPushButton("Cargar Imagen"); btn_c.clicked.connect(self.cargar_imagen)
        btn_w = QPushButton(" Usar Webcam"); btn_w.clicked.connect(self.usar_webcam)
        l_carga.addWidget(btn_c); l_carga.addWidget(btn_w); grupo_carga.setLayout(l_carga)
        
        grupo_modo = QGroupBox(" Modo")
        l_modo = QVBoxLayout(); btn_a = QPushButton(" Automático"); btn_a.clicked.connect(lambda: self.cambiar_modo(False))
        btn_m = QPushButton(" Manual"); btn_m.clicked.connect(lambda: self.cambiar_modo(True))
        l_modo.addWidget(btn_a); l_modo.addWidget(btn_m); grupo_modo.setLayout(l_modo)
        
        grupo_ajustes = QGroupBox(" Ajustes Canny")
        l_ajustes = QVBoxLayout()
        self.slider_canny1 = QSlider(Qt.Orientation.Horizontal); self.slider_canny1.setRange(0, 255); self.slider_canny1.setValue(50)
        self.slider_canny2 = QSlider(Qt.Orientation.Horizontal); self.slider_canny2.setRange(0, 255); self.slider_canny2.setValue(150)
        self.slider_canny1.valueChanged.connect(self.actualizar_escaner)
        self.slider_canny2.valueChanged.connect(self.actualizar_escaner)
        l_ajustes.addWidget(QLabel("Umbral 1:")); l_ajustes.addWidget(self.slider_canny1)
        l_ajustes.addWidget(QLabel("Umbral 2:")); l_ajustes.addWidget(self.slider_canny2)
        grupo_ajustes.setLayout(l_ajustes)
        
        grupo_acciones = QGroupBox(" Acciones")
        l_acc = QVBoxLayout()
        btn_e = QPushButton(" Re-Escanear"); btn_e.clicked.connect(self.escanear_documento)
        btn_m = QPushButton(" Blanco y Negro (Mejorar)"); btn_m.clicked.connect(self.mejorar_imagen)
        btn_g = QPushButton(" Guardar"); btn_g.clicked.connect(self.guardar_resultado)
        l_acc.addWidget(btn_e); l_acc.addWidget(btn_m); l_acc.addWidget(btn_g); grupo_acciones.setLayout(l_acc)
        
        layout_control.addWidget(grupo_carga); layout_control.addWidget(grupo_modo)
        layout_control.addWidget(grupo_ajustes); layout_control.addWidget(grupo_acciones)
        layout_control.addStretch(); layout.addWidget(panel_control, 1)

    def setup_menu(self):
        menubar = self.menuBar()
        archivo_menu = menubar.addMenu("&Archivo")
        abrir_action = QAction("&Abrir", self); abrir_action.triggered.connect(self.cargar_imagen)
        archivo_menu.addAction(abrir_action)

    def cargar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if archivo:
            self.imagen_original = cv2.imread(archivo)
            self.puntos_manuales = []
            self.mostrar_imagen(self.imagen_original, self.label_original)
            self.escanear_documento()

    def usar_webcam(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.imagen_original = frame
            self.puntos_manuales = []
            self.mostrar_imagen(self.imagen_original, self.label_original)
            self.escanear_documento()

    def cambiar_modo(self, manual):
        self.modo_manual = manual
        self.puntos_manuales = []
        msg = "Modo Manual: Haz clic en las 4 esquinas del documento en la imagen original." if manual else "Modo Automático activado."
        QMessageBox.information(self, "Modo de detección", msg)

    def mousePressEvent(self, event):
        # Lógica para detectar clics dentro del label_original
        if self.modo_manual and self.imagen_original is not None:
            # Convertir coordenadas globales a locales del label
            local_pos = self.label_original.mapFromParent(event.pos())
            
            if self.label_original.rect().contains(local_pos):
                # Mapear coordenadas del Label a la imagen original (ajustando el escalado)
                h, w, _ = self.imagen_original.shape
                lbl_w = self.label_original.width()
                lbl_h = self.label_original.height()
                
                # Cálculo de escala manteniendo aspecto
                scale = min(lbl_w/w, lbl_h/h)
                new_w, new_h = w * scale, h * scale
                offset_x = (lbl_w - new_w) / 2
                offset_y = (lbl_h - new_h) / 2
                
                img_x = int((local_pos.x() - offset_x) / scale)
                img_y = int((local_pos.y() - offset_y) / scale)
                
                if 0 <= img_x < w and 0 <= img_y < h:
                    self.puntos_manuales.append([img_x, img_y])
                    # Dibujar punto temporalmente
                    temp_img = self.imagen_original.copy()
                    for p in self.puntos_manuales:
                        cv2.circle(temp_img, (p[0], p[1]), 10, (0, 255, 0), -1)
                    self.mostrar_imagen(temp_img, self.label_original)
                    
                    if len(self.puntos_manuales) == 4:
                        self.escanear_documento()
                        self.puntos_manuales = [] # Reiniciar

    def ordenar_puntos(self, puntos):
        puntos = np.array(puntos).reshape(4, 2)
        ordenados = np.zeros((4, 2), dtype=np.float32)
        suma = puntos.sum(axis=1)
        ordenados[0] = puntos[np.argmin(suma)] # Top-left
        ordenados[2] = puntos[np.argmax(suma)] # Bottom-right
        diff = np.diff(puntos, axis=1)
        ordenados[1] = puntos[np.argmin(diff)] # Top-right
        ordenados[3] = puntos[np.argmax(diff)] # Bottom-left
        return ordenados

    def escanear_documento(self):
        if self.imagen_original is None: return
        
        pts_origen = None
        if self.modo_manual and len(self.puntos_manuales) == 0: # Si ya se procesaron los 4 puntos
             return # Esperar clics
            
        if not self.modo_manual:
            esquinas = self.detectar_documento_auto(self.imagen_original)
            if esquinas is not None:
                pts_origen = self.ordenar_puntos(esquinas)
            else:
                return # No avisar en cada cambio de slider para no molestar
        else:
            if len(self.puntos_manuales) == 4:
                pts_origen = self.ordenar_puntos(self.puntos_manuales)

        if pts_origen is not None:
            (tl, tr, br, bl) = pts_origen
            ancho = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))
            alto = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
            
            pts_destino = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
            M = cv2.getPerspectiveTransform(pts_origen, pts_destino)
            self.imagen_procesada = cv2.warpPerspective(self.imagen_original, M, (ancho, alto))
            self.mostrar_imagen(self.imagen_procesada, self.label_procesada)

    def detectar_documento_auto(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(desenfoque, self.slider_canny1.value(), self.slider_canny2.value())
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            c = max(contornos, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(aprox) == 4: return aprox
        return None

    def mejorar_imagen(self):
        if self.imagen_procesada is None: return
        gris = cv2.cvtColor(self.imagen_procesada, cv2.COLOR_BGR2GRAY)
        # Adaptive Threshold para efecto "papel limpio"
        mejora = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.imagen_procesada = cv2.cvtColor(mejora, cv2.COLOR_GRAY2BGR)
        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)

    def mostrar_imagen(self, imagen, label):
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def actualizar_escaner(self):
        if not self.modo_manual: self.escanear_documento()

    def guardar_resultado(self):
        if self.imagen_procesada is not None:
            nombre = f"scan_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(nombre, self.imagen_procesada)
            QMessageBox.information(self, "Éxito", f"Guardado como {nombre}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EscanerDocumentos()
    window.show()
    sys.exit(app.exec())
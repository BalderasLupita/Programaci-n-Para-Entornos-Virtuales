
import sys
import cv2
import cv2.aruco as aruco
import numpy as np
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QFileDialog, QMessageBox, QComboBox, QSlider)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class CamaraCalibrada(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📷 Cámara Calibrada - Capítulo 12")
        self.setGeometry(100, 100, 1400, 800)
        
        # Parámetros de cámara
        self.matriz_camara = None
        self.dist_coefs = None
        self.usar_calibracion = False
        
        # Captura de video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        # Para comparación
        self.frame_original = None
        self.frame_corregido = None
        self.modo_comparacion = "lado_a_lado"
        
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        
        # Panel de video
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        
        self.label_video = QLabel()
        self.label_video.setMinimumSize(900, 600)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        
        layout.addWidget(panel_video, 3)
        
        # Panel de control
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        # Grupo: Cargar calibración
        grupo_carga = QGroupBox("📂 Calibración")
        layout_carga = QVBoxLayout()
        
        btn_cargar = QPushButton("Cargar parámetros...")
        btn_cargar.clicked.connect(self.cargar_parametros)
        layout_carga.addWidget(btn_cargar)
        
        self.label_estado = QLabel("❌ No calibrada")
        self.label_estado.setStyleSheet("color: red; font-weight: bold;")
        layout_carga.addWidget(self.label_estado)
        
        grupo_carga.setLayout(layout_carga)
        layout_control.addWidget(grupo_carga)
        
        # Grupo: Modo de visualización
        grupo_modo = QGroupBox("👁️ Visualización")
        layout_modo = QVBoxLayout()
        
        self.combo_modo = QComboBox()
        self.combo_modo.addItems(["lado_a_lado", "deslizante", "comparacion_directa"])
        self.combo_modo.currentTextChanged.connect(
            lambda v: setattr(self, 'modo_comparacion', v))
        layout_modo.addWidget(self.combo_modo)
        
        self.cb_calibracion = QPushButton("🔧 Activar calibración")
        self.cb_calibracion.setCheckable(True)
        self.cb_calibracion.toggled.connect(self.toggle_calibracion)
        layout_modo.addWidget(self.cb_calibracion)
        
        grupo_modo.setLayout(layout_modo)
        layout_control.addWidget(grupo_modo)
        
        # Grupo: Información
        grupo_info = QGroupBox("ℹ️ Parámetros")
        layout_info = QVBoxLayout()
        
        self.info_text = QLabel(
            "Matriz cámara:\n--\n\n"
            "Distorsión:\n--\n\n"
            "Resolución: --"
        )
        layout_info.addWidget(self.info_text)
        
        grupo_info.setLayout(layout_info)
        layout_control.addWidget(grupo_info)
        
        # Grupo: Prueba con ArUco
        grupo_prueba = QGroupBox("🎯 Prueba AR")
        layout_prueba = QVBoxLayout()
        
        btn_probar_aruco = QPushButton("Probar con marcador ArUco")
        btn_probar_aruco.clicked.connect(self.probar_con_aruco)
        layout_prueba.addWidget(btn_probar_aruco)
        
        grupo_prueba.setLayout(layout_prueba)
        layout_control.addWidget(grupo_prueba)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
    
    def cargar_parametros(self):
        """Carga parámetros de calibración desde archivo"""
        archivo, _ = QFileDialog.getOpenFileName(
            self, "Cargar parámetros", "", "NPZ Files (*.npz);;JSON Files (*.json)")
        
        if archivo:
            try:
                if archivo.endswith('.npz'):
                    datos = np.load(archivo)
                    self.matriz_camara = datos['matriz_camara']
                    self.dist_coefs = datos['dist_coefs']
                else:
                    with open(archivo, 'r') as f:
                        datos = json.load(f)
                    self.matriz_camara = np.array(datos['matriz_camara'])
                    self.dist_coefs = np.array(datos['dist_coefs'])
                
                self.label_estado.setText("✅ Calibrada")
                self.label_estado.setStyleSheet("color: green; font-weight: bold;")
                self.actualizar_info()
                
                QMessageBox.information(self, "Éxito", 
                    "Parámetros cargados correctamente")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                    f"No se pudieron cargar los parámetros: {e}")
    
    def toggle_calibracion(self, activado):
        """Activa/desactiva el uso de calibración"""
        self.usar_calibracion = activado and self.matriz_camara is not None
        
        if activado and self.matriz_camara is None:
            self.cb_calibracion.setChecked(False)
            QMessageBox.warning(self, "Atención", 
                "Primero debes cargar los parámetros de calibración")
    
    def actualizar_info(self):
        """Actualiza la información mostrada"""
        if self.matriz_camara is not None:
            fx = self.matriz_camara[0, 0]
            fy = self.matriz_camara[1, 1]
            cx = self.matriz_camara[0, 2]
            cy = self.matriz_camara[1, 2]
            
            k1, k2, p1, p2, k3 = self.dist_coefs.ravel()[:5]
            
            info = f"Matriz cámara:\n"
            info += f"fx: {fx:.1f}, fy: {fy:.1f}\n"
            info += f"cx: {cx:.1f}, cy: {cy:.1f}\n\n"
            info += f"Distorsión:\n"
            info += f"k1: {k1:.3f}, k2: {k2:.3f}\n"
            info += f"p1: {p1:.3f}, p2: {p2:.3f}\n"
            info += f"k3: {k3:.3f}\n"
            
            self.info_text.setText(info)
    
    def corregir_frame(self, frame):
        """Aplica corrección de distorsión al frame"""
        if not self.usar_calibracion or self.matriz_camara is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Obtener nueva matriz de cámara
        nueva_matriz, roi = cv2.getOptimalNewCameraMatrix(
            self.matriz_camara, self.dist_coefs, (w, h), 1, (w, h))
        
        # Corregir distorsión
        frame_corregido = cv2.undistort(
            frame, self.matriz_camara, self.dist_coefs, None, nueva_matriz)
        
        # Recortar según ROI
        x, y, w, h = roi
        frame_corregido = frame_corregido[y:y+h, x:x+w]
        
        return frame_corregido
    
    def visualizar_comparacion(self, frame_original, frame_corregido):
        """Muestra la comparación según el modo seleccionado"""
        if frame_corregido is None:
            return frame_original
        
        h1, w1 = frame_original.shape[:2]
        h2, w2 = frame_corregido.shape[:2]
        
        if self.modo_comparacion == "lado_a_lado":
            # Redimensionar para que tengan la misma altura
            if h1 != h2:
                escala = h1 / h2
                nuevo_w2 = int(w2 * escala)
                frame_corregido = cv2.resize(frame_corregido, (nuevo_w2, h1))
            
            # Crear imagen combinada
            combinado = np.hstack([frame_original, frame_corregido])
            
            # Etiquetas
            cv2.putText(combinado, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combinado, "CORREGIDA", (w1 + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return combinado
        
        elif self.modo_comparacion == "comparacion_directa":
            # Mostrar original y corregida en ventanas separadas
            # (simplificado, mostramos una con toggle)
            if hasattr(self, '_mostrar_original'):
                self._mostrar_original = not self._mostrar_original
            else:
                self._mostrar_original = True
            
            if self._mostrar_original:
                cv2.putText(frame_original, "ORIGINAL", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame_original
            else:
                cv2.putText(frame_corregido, "CORREGIDA", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return frame_corregido
        
        else:  # deslizante
            # Crear línea divisoria móvil (simplificado)
            if not hasattr(self, '_pos_deslizante'):
                self._pos_deslizante = w1 // 2
            
            # Redimensionar corregida al tamaño de original
            frame_corregido_redim = cv2.resize(frame_corregido, (w1, h1))
            
            # Combinar con línea
            resultado = frame_original.copy()
            resultado[:, self._pos_deslizante:] = frame_corregido_redim[:, self._pos_deslizante:]
            
            # Dibujar línea
            cv2.line(resultado, (self._pos_deslizante, 0), 
                    (self._pos_deslizante, h1), (255, 255, 255), 3)
            
            return resultado
    
    def probar_con_aruco(self):
        """Prueba la calibración con detección de marcadores ArUco"""
        if self.matriz_camara is None:
            QMessageBox.warning(self, "Atención", 
                "Primero debes cargar los parámetros de calibración")
            return
        
        # Crear ventana de prueba
        cv2.namedWindow('Prueba ArUco con Calibración')
        
        diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parametros = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(diccionario, parametros)
        
        tamanio_marcador = 0.05
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Aplicar calibración
            frame_calibrado = self.corregir_frame(frame)
            
            # Detectar marcadores
            esquinas, ids, _ = detector.detectMarkers(frame_calibrado)
            
            if ids is not None:
                aruco.drawDetectedMarkers(frame_calibrado, esquinas, ids)
                
                for i in range(len(ids)):
                    # Estimar pose con parámetros calibrados
                    obj_points = np.array([[-tamanio_marcador/2, tamanio_marcador/2, 0],
                                           [tamanio_marcador/2, tamanio_marcador/2, 0],
                                           [tamanio_marcador/2, -tamanio_marcador/2, 0],
                                           [-tamanio_marcador/2, -tamanio_marcador/2, 0]], 
                                          dtype=np.float32)
                    
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, esquinas[i][0], 
                        self.matriz_camara, self.dist_coefs)
                    
                    if success:
                        # Dibujar ejes
                        cv2.drawFrameAxes(frame_calibrado, self.matriz_camara, 
                                         self.dist_coefs, rvec, tvec, 0.03)
                        
                        # Mostrar distancia
                        distancia = np.linalg.norm(tvec)
                        cv2.putText(frame_calibrado, f"Dist: {distancia:.2f}m", 
                                   (10, 30 + i*30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Prueba ArUco con Calibración', frame_calibrado)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow('Prueba ArUco con Calibración')
    
    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.frame_original = frame.copy()
        self.frame_corregido = self.corregir_frame(frame) if self.usar_calibracion else None
        
        # Visualizar según modo
        if self.usar_calibracion and self.frame_corregido is not None:
            frame_mostrar = self.visualizar_comparacion(
                self.frame_original, self.frame_corregido)
        else:
            frame_mostrar = self.frame_original
            if not self.usar_calibracion:
                cv2.putText(frame_mostrar, "SIN CALIBRAR", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        self.mostrar_imagen(frame_mostrar)
    
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
    ventana = CamaraCalibrada()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


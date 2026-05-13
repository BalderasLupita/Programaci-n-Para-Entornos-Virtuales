import sys
import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
# Se importa el módulo vision y los componentes necesarios para la detección de manos
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QSlider, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class CuboARInteractivo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎲 Cubo AR Interactivo - MediaPipe Vision")
        self.setGeometry(100, 100, 1400, 800)
        
        # Configurar detector ArUco
        self.diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parametros = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.diccionario, self.parametros)
        
        # Parámetros de cámara
        self.matriz_camara = np.array([[1000, 0, 640],
                                       [0, 1000, 360],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coefs = np.zeros((4, 1))
        
        # Tamaños
        self.tamanio_marcador = 0.05
        self.lado_cubo = 0.03
        self.cubo_3d = self.crear_cubo_3d(self.lado_cubo)
        self.caras_cubo = self.definir_caras_cubo()
        
        # Variables de interacción
        self.rotacion_x = 0
        self.rotacion_y = 0
        self.rotacion_auto = True
        self.color_actual = (0, 255, 0)
        self.modo_color = "solido"

        # --- CONFIGURACIÓN DE MEDIAPIPE VISION ---
        # Nota: Asegúrate de tener el archivo 'hand_landmarker.task' en la misma carpeta
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            running_mode=vision.RunningMode.IMAGE
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Captura de video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        self.setup_ui()
    
    def crear_cubo_3d(self, lado):
        l = lado / 2
        return np.float32([
            [-l, -l, 0], [l, -l, 0], [l, l, 0], [-l, l, 0],
            [-l, -l, l], [l, -l, l], [l, l, l], [-l, l, l]
        ])
    
    def definir_caras_cubo(self):
        return [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]
        ]
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        panel_video = QWidget()
        layout_video = QVBoxLayout(panel_video)
        self.label_video = QLabel()
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_video.addWidget(self.label_video)
        layout.addWidget(panel_video, 3)
        
        panel_control = QWidget()
        panel_control.setMaximumWidth(350)
        layout_control = QVBoxLayout(panel_control)
        
        grupo_rotacion = QGroupBox("🔄 Rotación")
        layout_rotacion = QVBoxLayout()
        btn_auto = QPushButton("Auto/Manual")
        btn_auto.setCheckable(True)
        btn_auto.setChecked(True)
        btn_auto.toggled.connect(lambda v: setattr(self, 'rotacion_auto', v))
        layout_rotacion.addWidget(btn_auto)
        
        layout_rotacion.addWidget(QLabel("Rotación X:"))
        slider_rx = QSlider(Qt.Orientation.Horizontal)
        slider_rx.setRange(-180, 180)
        slider_rx.valueChanged.connect(lambda v: setattr(self, 'rotacion_x', v))
        layout_rotacion.addWidget(slider_rx)
        
        layout_rotacion.addWidget(QLabel("Rotación Y:"))
        slider_ry = QSlider(Qt.Orientation.Horizontal)
        slider_ry.setRange(-180, 180)
        slider_ry.valueChanged.connect(lambda v: setattr(self, 'rotacion_y', v))
        layout_rotacion.addWidget(slider_ry)
        grupo_rotacion.setLayout(layout_rotacion)
        layout_control.addWidget(grupo_rotacion)
        
        grupo_color = QGroupBox("🎨 Color")
        layout_color = QVBoxLayout()
        self.combo_color = QComboBox()
        self.combo_color.addItems(["sólido", "arcoíris", "por cara", "distancia"])
        self.combo_color.currentTextChanged.connect(lambda v: setattr(self, 'modo_color', v))
        layout_color.addWidget(self.combo_color)
        grupo_color.setLayout(layout_color)
        layout_control.addWidget(grupo_color)
        
        layout_control.addStretch()
        layout.addWidget(panel_control, 1)
    
    def detectar_gesto(self, hand_landmarks):
        if not hand_landmarks:
            return "none"
        
        # En mp.vision, los landmarks se acceden como una lista directa
        p_indice = hand_landmarks[8]
        p_medio = hand_landmarks[12]
        p_anular = hand_landmarks[16]
        p_menique = hand_landmarks[20]
        n_indice = hand_landmarks[6]
        
        if (p_indice.y < n_indice.y and p_medio.y > n_indice.y and
            p_anular.y > n_indice.y and p_menique.y > n_indice.y):
            return "indice"
        elif (p_indice.y < n_indice.y and p_medio.y < n_indice.y and
              p_anular.y > n_indice.y and p_menique.y > n_indice.y):
            return "paz"
        elif all(p.y < n_indice.y for p in [p_indice, p_medio, p_anular, p_menique]):
            return "abierta"
        return "none"

    def dibujar_cubo_coloreado(self, img, puntos_2d):
        puntos = np.int32(puntos_2d).reshape(-1, 2)
        if self.modo_color == "sólido":
            overlay = img.copy()
            for cara in self.caras_cubo:
                pts_cara = np.array([puntos[i] for i in cara], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts_cara], self.color_actual)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            self.dibujar_aristas(img, puntos, (255, 255, 255), 2)
        elif self.modo_color == "arcoíris":
            colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            for i, cara in enumerate(self.caras_cubo):
                pts_cara = np.array([puntos[j] for j in cara], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts_cara], colores[i % len(colores)])
            self.dibujar_aristas(img, puntos, (255, 255, 255), 1)

    def dibujar_aristas(self, img, puntos, color, grosor):
        aristas = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for a, b in aristas:
            cv2.line(img, tuple(puntos[a]), tuple(puntos[b]), color, grosor)

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        # Procesamiento con mediapipe.vision
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.hand_landmarker.detect(mp_image)
        
        gesto = "none"
        if detection_result.hand_landmarks:
            # Obtenemos los landmarks de la primera mano detectada
            landmarks = detection_result.hand_landmarks[0]
            gesto = self.detectar_gesto(landmarks)
            
            # (Opcional) El dibujo de landmarks manual o mediante utilidades antiguas
            # Para brevedad, omitimos el dibujo de esqueletos de mano en esta versión vision pura
        
        if gesto == "paz":
            self.rotacion_auto = False
            self.rotacion_x += 2
        elif gesto == "indice":
            self.modo_color = "arcoíris"
        elif gesto == "abierta":
            self.modo_color = "sólido"
            self.color_actual = (0, 255, 0)
        
        esquinas, ids, _ = self.detector.detectMarkers(frame)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, esquinas, ids)
            for i in range(len(ids)):
                obj_points = np.array([[-self.tamanio_marcador/2, self.tamanio_marcador/2, 0],
                                       [self.tamanio_marcador/2, self.tamanio_marcador/2, 0],
                                       [self.tamanio_marcador/2, -self.tamanio_marcador/2, 0],
                                       [-self.tamanio_marcador/2, -self.tamanio_marcador/2, 0]], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(obj_points, esquinas[i][0], self.matriz_camara, self.dist_coefs)
                if success:
                    if self.rotacion_auto:
                        self.rotacion_y = (self.rotacion_y + 2) % 360
                    
                    R_extra, _ = cv2.Rodrigues(np.array([self.rotacion_x * np.pi/180, self.rotacion_y * np.pi/180, 0]))
                    cubo_rotado = np.dot(self.cubo_3d, R_extra.T)
                    imgpts, _ = cv2.projectPoints(cubo_rotado, rvec, tvec, self.matriz_camara, self.dist_coefs)
                    self.dibujar_cubo_coloreado(frame, imgpts)
        
        self.mostrar_imagen(frame)
    
    def mostrar_imagen(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label_video.setPixmap(pixmap)
    
    def closeEvent(self, event):
        self.cap.release()
        self.hand_landmarker.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ventana = CuboARInteractivo()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
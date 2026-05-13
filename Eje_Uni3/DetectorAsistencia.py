import sys
import cv2
import numpy as np
import json
import os
import urllib.request
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QListWidget, QListWidgetItem, QInputDialog,
                             QMessageBox, QLineEdit, QGridLayout)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QImage, QPixmap

class DetectorAsistencia(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("👥 Detector de Asistencia - OpenCV + PyQt6")
        self.setGeometry(100, 100, 1200, 700)
        
        # --- Configuración de Rutas ---
        self.ruta_base = os.path.dirname(os.path.abspath(__file__))
        self.prototxt = os.path.join(self.ruta_base, "deploy.prototxt")
        self.modelo = os.path.join(self.ruta_base, "res10_300x300_ssd_iter_140000.caffemodel")
        self.archivo_bd = os.path.join(self.ruta_base, "personas.json")

        # Variables de cámara
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)
        
        # Inicialización
        self.cargar_modelo_dnn()
        self.personas_conocidas = self.cargar_personas()
        self.confianza_minima = 0.5
        self.setup_ui()
        
    def cargar_modelo_dnn(self):
        """Descarga y carga la red neuronal para detección de rostros"""
        try:
            if not os.path.exists(self.prototxt) or not os.path.exists(self.modelo):
                print("Descargando modelos... por favor espera.")
                url_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                url_model = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                urllib.request.urlretrieve(url_proto, self.prototxt)
                urllib.request.urlretrieve(url_model, self.modelo)
            
            self.red_dnn = cv2.dnn.readNetFromCaffe(self.prototxt, self.modelo)
            self.usar_dnn = True
        except Exception as e:
            print(f"Error cargando red neuronal: {e}")
            self.usar_dnn = False
    
    def cargar_personas(self):
        """Lee la base de datos manejando posibles errores de formato (JSON Corrupto)"""
        if os.path.exists(self.archivo_bd):
            try:
                with open(self.archivo_bd, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                print("Archivo personas.json dañado. Reiniciando base de datos...")
                return {}
        return {}
    
    def guardar_personas(self):
        """Guarda los datos en el archivo JSON"""
        with open(self.archivo_bd, 'w') as f:
            json.dump(self.personas_conocidas, f, indent=2)

    def setup_ui(self):
        """Configuración de la interfaz gráfica"""
        central = QWidget()
        self.setCentralWidget(central)
        layout_principal = QHBoxLayout(central)
        
        # Panel de Video
        panel_video = QWidget()
        layout_v = QVBoxLayout(panel_video)
        self.label_video = QLabel()
        self.label_video.setStyleSheet("border: 2px solid #555; background: black;")
        self.label_video.setMinimumSize(640, 480)
        layout_v.addWidget(self.label_video)
        
        self.info_label = QLabel("Iniciando cámara...")
        layout_v.addWidget(self.info_label)
        layout_principal.addWidget(panel_video, 3)
        
        # Panel de Control
        panel_ctrl = QWidget()
        panel_ctrl.setMaximumWidth(300)
        layout_c = QVBoxLayout(panel_ctrl)
        
        # Lista de asistencia
        grupo_asist = QGroupBox("📋 Rostros en Cámara")
        lay_a = QVBoxLayout()
        self.lista_asistencia = QListWidget()
        lay_a.addWidget(self.lista_asistencia)
        btn_reg = QPushButton("➕ Registrar Persona")
        btn_reg.clicked.connect(self.registrar_persona)
        lay_a.addWidget(btn_reg)
        grupo_asist.setLayout(lay_a)
        layout_c.addWidget(grupo_asist)
        
        # Lista de BD
        grupo_bd = QGroupBox("💾 Base de Datos")
        lay_b = QVBoxLayout()
        self.lista_bd = QListWidget()
        self.actualizar_lista_bd()
        lay_b.addWidget(self.lista_bd)
        btn_del = QPushButton("🗑️ Eliminar Seleccionado")
        btn_del.clicked.connect(self.eliminar_persona)
        lay_b.addWidget(btn_del)
        grupo_bd.setLayout(lay_b)
        layout_c.addWidget(grupo_bd)

        layout_principal.addWidget(panel_ctrl, 1)

    def detectar_rostros(self, frame):
        """Procesa el frame para encontrar rostros con DNN"""
        h, w = frame.shape[:2]
        if not self.usar_dnn: return []
        
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.red_dnn.setInput(blob)
        detecciones = self.red_dnn.forward()
        
        rostros = []
        for i in range(detecciones.shape[2]):
            conf = detecciones[0, 0, i, 2]
            if conf > self.confianza_minima:
                box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                rostros.append((max(0, x1), max(0, y1), x2-x1, y2-y1, conf))
        return rostros

    def reconocer_persona(self, w, h):
        """Compara dimensiones para 'reconocer' (Lógica simplificada)"""
        for nombre, datos in self.personas_conocidas.items():
            if abs(w - datos['w']) < 35 and abs(h - datos['h']) < 35:
                return nombre
        return "Desconocido"

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        rostros = self.detectar_rostros(frame)
        presentes = []
        
        for (x, y, w, h, conf) in rostros:
            nombre = self.reconocer_persona(w, h)
            presentes.append(nombre)
            
            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{nombre}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        self.actualizar_ui_asistencia(presentes)
        self.mostrar_en_label(frame)

    def actualizar_ui_asistencia(self, presentes):
        self.lista_asistencia.clear()
        for p in set(presentes):
            item = QListWidgetItem(p)
            item.setForeground(Qt.GlobalColor.green if p != "Desconocido" else Qt.GlobalColor.red)
            self.lista_asistencia.addItem(item)
        self.info_label.setText(f"Detecciones: {len(presentes)}")

    def registrar_persona(self):
        nombre, ok = QInputDialog.getText(self, "Nuevo Registro", "Nombre:")
        if ok and nombre:
            ret, frame = self.cap.read()
            rostros = self.detectar_rostros(frame)
            if rostros:
                _, _, w, h, _ = rostros[0]
                self.personas_conocidas[nombre] = {"w": int(w), "h": int(h)}
                self.guardar_personas()
                self.actualizar_lista_bd()
                QMessageBox.information(self, "OK", f"{nombre} registrado.")
            else:
                QMessageBox.warning(self, "Error", "No se detecta rostro.")

    def eliminar_persona(self):
        item = self.lista_bd.currentItem()
        if item:
            nombre = item.text()
            del self.personas_conocidas[nombre]
            self.guardar_personas()
            self.actualizar_lista_bd()

    def actualizar_lista_bd(self):
        self.lista_bd.clear()
        self.lista_bd.addItems(self.personas_conocidas.keys())

    def mostrar_en_label(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.label_video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DetectorAsistencia()
    win.show()
    sys.exit(app.exec())
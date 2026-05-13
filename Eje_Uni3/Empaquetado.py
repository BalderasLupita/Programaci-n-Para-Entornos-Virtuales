import os
import sys
import cv2
import numpy as np
import psutil
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QGridLayout, QProgressBar, QCheckBox, QSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen

def recurso_path(relative_path):
    """ Obtiene la ruta absoluta para recursos, compatible con PyInstaller """
    try:
        # PyInstaller crea una carpeta temporal y guarda la ruta en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Busca la ruta absoluta de la carpeta donde está el script
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class MonitorRendimiento(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 Dashboard de Performance - Empaquetado")
        self.setGeometry(100, 100, 1200, 700)
        
        # Historial
        self.historial_fps = deque(maxlen=100)
        self.historial_cpu = deque(maxlen=100)
        
        # Configuración
        self.proceso_actual = psutil.Process()
        self.mostrar_graficos = True
        
        self.setup_ui()
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_todo)
        self.timer.start(100)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Título
        titulo = QLabel("MONITOR DE SISTEMA")
        titulo.setStyleSheet("font-size: 20px; font-weight: bold;")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(titulo)

        # Grid de métricas
        grid = QGridLayout()
        
        # CPU Label y Progreso
        self.grupo_cpu = QGroupBox("Utilización CPU")
        ly_cpu = QVBoxLayout()
        self.label_cpu = QLabel("0%")
        self.label_cpu.setStyleSheet("font-size: 18px;")
        ly_cpu.addWidget(self.label_cpu)
        self.grupo_cpu.setLayout(ly_cpu)
        grid.addWidget(self.grupo_cpu, 0, 0)

        layout.addLayout(grid)

        # Área de Gráficos (Debe ser QLabel para setPixmap)
        self.widget_graficos = QLabel()
        self.widget_graficos.setMinimumHeight(300)
        self.widget_graficos.setStyleSheet("border: 1px solid #555; background: #000;")
        layout.addWidget(self.widget_graficos)

        # Botón de Reset
        btn_reset = QPushButton("Resetear Datos")
        btn_reset.clicked.connect(self.limpiar_datos)
        layout.addWidget(btn_reset)

    def actualizar_todo(self):
        # Obtener CPU
        cpu = self.proceso_actual.cpu_percent()
        self.label_cpu.setText(f"{cpu}%")
        self.historial_cpu.append(cpu)
        
        # Simular FPS
        fps = 60 + np.random.randint(-5, 5)
        self.historial_fps.append(fps)
        
        if self.mostrar_graficos:
            self.dibujar_graficos()

    def dibujar_graficos(self):
        if self.widget_graficos.width() <= 0: return

        pixmap = QPixmap(self.widget_graficos.size())
        pixmap.fill(QColor(20, 20, 20))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        ancho = self.widget_graficos.width()
        alto = self.widget_graficos.height()

        def dibujar_tendencia(datos, color):
            if len(datos) < 2: return
            puntos = []
            max_d = max(datos) if max(datos) > 0 else 100
            for i, v in enumerate(datos):
                x = int((i / 100) * ancho)
                y = int(alto - (v / max_d) * (alto - 40) - 20)
                puntos.append((x, y))
            
            painter.setPen(QPen(color, 2))
            for i in range(len(puntos)-1):
                painter.drawLine(puntos[i][0], puntos[i][1], puntos[i+1][0], puntos[i+1][1])

        dibujar_tendencia(list(self.historial_cpu), QColor(255, 50, 50))
        dibujar_tendencia(list(self.historial_fps), QColor(50, 255, 50))
        
        painter.end()
        self.widget_graficos.setPixmap(pixmap)

    def limpiar_datos(self):
        self.historial_cpu.clear()
        self.historial_fps.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Ejemplo de cómo usarías recurso_path para un icono o estilo
    # ruta_icon = recurso_path("icono.png") 
    
    ventana = MonitorRendimiento()
    ventana.show()
    sys.exit(app.exec())
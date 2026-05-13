import sys
import cv2
import numpy as np
import time
import psutil
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QGridLayout, QProgressBar, QCheckBox, QSpinBox,
                             QComboBox)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen

class MonitorRendimiento(QMainWindow):
    """Dashboard de monitoreo de rendimiento en tiempo real"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 Dashboard de Performance - Capítulo 17")
        self.setGeometry(100, 100, 1400, 800)
        
        self.historial_fps = deque(maxlen=100)
        self.historial_cpu = deque(maxlen=100)
        self.historial_latencia = deque(maxlen=100)
        self.historial_memoria = deque(maxlen=100)
        
        self.mostrar_graficos = True
        self.intervalo_actualizacion = 100
        self.proceso_actual = psutil.Process()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_metricas)
        self.timer.start(self.intervalo_actualizacion)
        
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        titulo = QLabel("📊 MONITOR DE RENDIMIENTO EN TIEMPO REAL")
        titulo.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(titulo)
        
        grid = QGridLayout()
        
        # FPS
        grupo_fps = self.crear_grupo_metrica("🎮 FPS", "0")
        self.label_fps = grupo_fps.findChild(QLabel, "valor")
        self.progress_fps = QProgressBar()
        self.progress_fps.setRange(0, 60)
        grupo_fps.layout().addWidget(self.progress_fps)
        grid.addWidget(grupo_fps, 0, 0)
        
        # CPU
        grupo_cpu = self.crear_grupo_metrica("⚙️ CPU", "0%")
        self.label_cpu = grupo_cpu.findChild(QLabel, "valor")
        self.progress_cpu = QProgressBar()
        self.progress_cpu.setRange(0, 100)
        grupo_cpu.layout().addWidget(self.progress_cpu)
        grid.addWidget(grupo_cpu, 0, 1)
        
        # Memoria
        grupo_mem = self.crear_grupo_metrica("💾 Memoria", "0 MB")
        self.label_mem = grupo_mem.findChild(QLabel, "valor")
        self.progress_mem = QProgressBar()
        self.progress_mem.setRange(0, 100)
        grupo_mem.layout().addWidget(self.progress_mem)
        grid.addWidget(grupo_mem, 0, 2)
        
        # Latencia
        grupo_lat = self.crear_grupo_metrica("⏱️ Latencia", "0 ms")
        self.label_lat = grupo_lat.findChild(QLabel, "valor")
        self.progress_lat = QProgressBar()
        self.progress_lat.setRange(0, 100)
        grupo_lat.layout().addWidget(self.progress_lat)
        grid.addWidget(grupo_lat, 0, 3)
        
        layout.addLayout(grid)
        
        # --- CAMBIO CLAVE AQUÍ ---
        # Cambiamos QWidget por QLabel para que soporte setPixmap
        self.widget_graficos = QLabel() 
        self.widget_graficos.setMinimumHeight(300)
        self.widget_graficos.setStyleSheet("border: 1px solid #333; background-color: #1e1e1e;")
        self.widget_graficos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.widget_graficos)
        # -------------------------

        controles = QHBoxLayout()
        self.cb_graficos = QCheckBox("Mostrar gráficos")
        self.cb_graficos.setChecked(True)
        self.cb_graficos.toggled.connect(self.toggle_graficos)
        controles.addWidget(self.cb_graficos)
        
        controles.addWidget(QLabel("Intervalo (ms):"))
        self.spin_intervalo = QSpinBox()
        self.spin_intervalo.setRange(50, 1000)
        self.spin_intervalo.setValue(100)
        self.spin_intervalo.valueChanged.connect(self.cambiar_intervalo)
        controles.addWidget(self.spin_intervalo)
        
        controles.addStretch()
        
        btn_reset = QPushButton("🔄 Reset histórico")
        btn_reset.clicked.connect(self.reset_historial)
        controles.addWidget(btn_reset)
        
        layout.addLayout(controles)
        
        grupo_procesos = QGroupBox("📋 Procesos del sistema")
        layout_procesos = QVBoxLayout()
        self.tabla_procesos = QLabel("Recopilando información...")
        layout_procesos.addWidget(self.tabla_procesos)
        grupo_procesos.setLayout(layout_procesos)
        layout.addWidget(grupo_procesos)

    def crear_grupo_metrica(self, titulo, valor_inicial):
        grupo = QGroupBox(titulo)
        layout = QVBoxLayout()
        label_valor = QLabel(valor_inicial)
        label_valor.setObjectName("valor")
        label_valor.setStyleSheet("font-size: 24px; font-weight: bold;")
        label_valor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_valor)
        grupo.setLayout(layout)
        return grupo

    def actualizar_metricas(self):
        try:
            cpu_percent = self.proceso_actual.cpu_percent()
            self.label_cpu.setText(f"{cpu_percent:.1f}%")
            self.progress_cpu.setValue(int(cpu_percent))
            self.historial_cpu.append(cpu_percent)
            
            mem_info = self.proceso_actual.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            self.label_mem.setText(f"{mem_mb:.1f} MB")
            mem_percent = (mem_info.rss / psutil.virtual_memory().total) * 100
            self.progress_mem.setValue(int(min(mem_percent * 10, 100))) # Escala visual
            self.historial_memoria.append(mem_mb)
            
            fps = 30 + np.random.randn() * 5
            self.label_fps.setText(f"{fps:.1f}")
            self.progress_fps.setValue(int(fps))
            self.historial_fps.append(fps)
            
            latencia = 33 + np.random.randn() * 10
            self.label_lat.setText(f"{latencia:.1f} ms")
            self.progress_lat.setValue(int(min(latencia, 100)))
            self.historial_latencia.append(latencia)
            
            if self.mostrar_graficos:
                self.actualizar_graficos()
            self.actualizar_procesos()
        except Exception as e:
            print(f"Error en métricas: {e}")

    def actualizar_graficos(self):
        if not self.widget_graficos.width() > 0: return
        
        pixmap = QPixmap(self.widget_graficos.size())
        pixmap.fill(QColor(30, 30, 30))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        ancho = self.widget_graficos.width()
        alto = self.widget_graficos.height()
        
        # Grid
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(0, ancho, 50): painter.drawLine(i, 0, i, alto)
        for i in range(0, alto, 50): painter.drawLine(0, i, ancho, i)
        
        def dibujar_linea(datos, color):
            if len(datos) < 2: return
            puntos = []
            max_val = max(datos) if max(datos) > 0 else 1
            for i, val in enumerate(datos):
                x = int((i / 100) * ancho)
                y = int(alto - (val / max_val) * (alto - 50) - 25)
                puntos.append((x, y))
            
            painter.setPen(QPen(color, 2))
            for i in range(len(puntos) - 1):
                painter.drawLine(puntos[i][0], puntos[i][1], puntos[i+1][0], puntos[i+1][1])
        
        dibujar_linea(list(self.historial_fps), QColor(0, 255, 0))
        dibujar_linea(list(self.historial_cpu), QColor(255, 0, 0))
        dibujar_linea(list(self.historial_latencia), QColor(255, 255, 0))
        painter.end()
        
        self.widget_graficos.setPixmap(pixmap)

    def actualizar_procesos(self):
        procesos = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                info = proc.info
                if info['cpu_percent'] > 0.5:
                    procesos.append(f"PID {info['pid']}: {info['name'][:15]} | CPU: {info['cpu_percent']}%")
            self.tabla_procesos.setText("\n".join(procesos[:10]))
        except: pass

    def toggle_graficos(self, activado):
        self.mostrar_graficos = activado
        if not activado: self.widget_graficos.clear()

    def cambiar_intervalo(self, intervalo):
        self.timer.setInterval(intervalo)

    def reset_historial(self):
        self.historial_fps.clear()
        self.historial_cpu.clear()
        self.historial_latencia.clear()
        self.historial_memoria.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = MonitorRendimiento()
    ventana.show()
    sys.exit(app.exec())
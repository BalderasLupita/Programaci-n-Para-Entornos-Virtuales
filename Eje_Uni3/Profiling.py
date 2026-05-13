import cv2
import numpy as np
import time
from collections import deque

# --- CLASE PROFILER (Métrica de Rendimiento) ---
class ARProfiler:
    def __init__(self, ventana_tiempo=30):
        self.ventana_tiempo = ventana_tiempo
        self.metricas = {
            'captura': deque(maxlen=ventana_tiempo),
            'deteccion': deque(maxlen=ventana_tiempo),
            'render': deque(maxlen=ventana_tiempo),
            'total': deque(maxlen=ventana_tiempo)
        }
    
    def tic(self, etapa):
        self.metricas[etapa].append(time.time())
    
    def toc(self, etapa):
        if len(self.metricas[etapa]) > 0:
            inicio = self.metricas[etapa][-1]
            duracion = time.time() - inicio
            self.metricas[etapa][-1] = duracion
            return duracion
        return 0
    
    def get_tiempo_promedio(self, etapa):
        if len(self.metricas[etapa]) > 0:
            return sum(self.metricas[etapa]) / len(self.metricas[etapa])
        return 0
    
    def get_fps(self):
        if len(self.metricas['total']) > 0:
            tiempo_total = sum(self.metricas['total'])
            return len(self.metricas['total']) / tiempo_total
        return 0
    
    def get_reporte(self):
        promedio_total = self.get_tiempo_promedio('total')
        reporte = "\n📊 REPORTE DE RENDIMIENTO\n" + "="*30 + "\n"
        for etapa in ['captura', 'deteccion', 'render']:
            t_ms = self.get_tiempo_promedio(etapa) * 1000
            porc = (t_ms / (promedio_total * 1000)) * 100 if promedio_total > 0 else 0
            reporte += f"{etapa.capitalize():10}: {t_ms:6.2f} ms ({porc:5.1f}%)\n"
        reporte += f"FPS: {self.get_fps():.1f}\n"
        return reporte

    def dibujar_grafico(self, frame, ancho=200, alto=100):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - ancho - 10, 10), (w - 10, 10 + alto), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        colores = {'captura': (255, 0, 0), 'deteccion': (0, 255, 0), 'render': (0, 0, 255)}
        x_base = w - ancho - 5
        
        for i, etapa in enumerate(['captura', 'deteccion', 'render']):
            if len(self.metricas[etapa]) > 1:
                tiempos = list(self.metricas[etapa])
                max_t = max(max(tiempos), 0.001)
                for j in range(1, len(tiempos)):
                    y1 = int(10 + alto - 10 - (tiempos[j-1]/max_t * (alto-20)))
                    y2 = int(10 + alto - 10 - (tiempos[j]/max_t * (alto-20)))
                    cv2.line(frame, (x_base + j*5, y1), (x_base + (j+1)*5, y2), colores[etapa], 1)
        
        cv2.putText(frame, f"FPS: {self.get_fps():.1f}", (w - ancho, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

# --- CLASE APLICACIÓN ---
class AppAROptimizable:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.profiler = ARProfiler()
        self.frame_skip = 0
        self.resolucion_procesamiento = (640, 360)
        self.calidad_render = 'alta'
    
    def procesar_frame(self, frame):
        # Simula una carga de CPU (ej. MediaPipe o IA)
        time.sleep(0.015) 
        return frame
    
    def renderizar(self, frame):
        if self.calidad_render == 'alta':
            time.sleep(0.010) # Simula render complejo
            for i in range(15):
                cv2.circle(frame, (50 + i*40, 240), 25, (0, 255, 255), 2)
        else:
            time.sleep(0.002)
            cv2.circle(frame, (320, 240), 50, (0, 255, 0), -1)
        return frame
    
    def run(self):
        while True:
            self.profiler.tic('total')
            
            self.profiler.tic('captura')
            ret, frame = self.cap.read()
            self.profiler.toc('captura')
            
            if not ret: break
            
            # 1. Optimización: Redimensionar ANTES de detectar
            frame_proc = cv2.resize(frame, self.resolucion_procesamiento)
            
            self.profiler.tic('deteccion')
            frame_proc = self.procesar_frame(frame_proc)
            self.profiler.toc('deteccion')
            
            self.profiler.tic('render')
            frame = self.renderizar(frame)
            self.profiler.toc('render')
            
            self.profiler.toc('total')
            frame = self.profiler.dibujar_grafico(frame)
            
            # UI de estado
            cv2.putText(frame, f"Res: {self.resolucion_procesamiento}", (10, 30), 1, 1, (255,255,255), 1)
            cv2.putText(frame, f"Render: {self.calidad_render}", (10, 55), 1, 1, (255,255,255), 1)
            cv2.imshow('AR Profiling', frame)
            
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('1'): self.resolucion_procesamiento = (320, 180)
            elif tecla == ord('2'): self.resolucion_procesamiento = (640, 360)
            elif tecla == ord('3'): self.calidad_render = 'baja'
            elif tecla == ord('4'): self.calidad_render = 'alta'
            elif tecla == ord('q'): break
            
            if self.frame_skip % 30 == 0:
                print(self.profiler.get_reporte())
            self.frame_skip += 1

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    AppAROptimizable().run()
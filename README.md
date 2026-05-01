# Proyecto de Visión Artificial con OpenCV y MediaPipe

## 👤 Información del Estudiante
* **Nombre:** Maria Guadalupe Balderas Cortez
* **Grupo:** GIEV3081-E
* **Plataforma de Integración:** Unity (vía Python/OpenCV)

---

## 📝 Descripción del Proyecto
Este proyecto consiste en una serie de herramientas y algoritmos desarrollados para la detección, procesamiento y análisis de imágenes en tiempo real. El objetivo principal es extraer información del mundo físico (como colores, formas y gestos biométricos) para su integración y uso dentro de entornos interactivos en Unity.

A través de este sistema, se logra un puente entre la captura de video y la lógica de juego, permitiendo que acciones del mundo real controlen elementos virtuales.

## 🚀 Características Principales

### 👁️ Visión por Computadora (OpenCV)
* **Procesamiento de Espacios de Color:** Segmentación y aislamiento de canales BGR y HSV para rastreo de objetos.
* **Análisis Geométrico:** Detección de contornos y clasificación de figuras (triángulos, rectángulos, círculos) con sistema de conteo.
* **Filtros e Interacción:** Manipulación de matrices numéricas para modificar brillo, contraste y saturación.

### 🧠 Inteligencia Artificial y Biometría (MediaPipe)
* **Detección de Parpadeo (EAR):** Implementación del *Eye Aspect Ratio* para medir la apertura ocular y detectar fatiga o parpadeos.
* **Reconocimiento de Gestos:** Seguimiento de puntos clave (*landmarks*) en las manos para interpretar comandos gestuales.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.x
* **Librerías:** * `OpenCV`: Procesamiento de imagen y video.
    * `MediaPipe`: Modelos de Machine Learning para biometría.
    * `NumPy`: Manejo de matrices y cálculos matemáticos.
* **Motor de Juego:** Unity (Integración de datos).

---
*Este proyecto forma parte de la formación académica en el área de Gráficos e Interacción en Entornos Virtuales.*

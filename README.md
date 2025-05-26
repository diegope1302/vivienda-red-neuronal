# 🧠 Predicción de Precios de Viviendas con Redes Neuronales y Regresión Lineal

Este proyecto utiliza técnicas de aprendizaje automático para predecir los precios de viviendas en California a partir de un conjunto de datos real, aplicando tanto un modelo de regresión lineal simple como una red neuronal con TensorFlow/Keras.

---

## 🎯 Objetivo

Investigar y aplicar modelos de aprendizaje automático supervisado para predecir precios de viviendas, integrando:

- Fundamentos de regresión lineal y redes neuronales artificiales.
- Programación en Python y uso de bibliotecas especializadas.
- Control de versiones con GitHub.
- Visualización de resultados con gráficos.

---

## ⚙️ Metodología

1. **Cargar el dataset**: Se usó el conjunto de datos `California Housing` de la librería `scikit-learn`.

2. **Preprocesamiento**:
   - Normalización de datos con `StandardScaler`.
   - División de los datos en entrenamiento (80%) y prueba (20%).

3. **Modelos implementados**:
   - **Regresión lineal simple**: Se predijo el valor medio de una vivienda usando una sola variable (`AveRooms`).
   - **Red neuronal con Keras**:
     - Entrada: 8 variables del dataset.
     - Capas ocultas: 2 capas densas con activación ReLU.
     - Capa de salida: 1 neurona para regresión.
     - Entrenamiento: 50 épocas con optimizador Adam y función de pérdida `MSE`.

4. **Visualización**:
   - Se generaron gráficos para comparar valores reales y valores predichos.
   - Se mostraron los resultados en subplots para facilitar la comparación.

5. **Control de versiones**:
   - El código fuente y resultados se gestionaron mediante un repositorio GitHub.

---

## 📊 Resultados

- El modelo de regresión lineal logró representar una tendencia básica de los datos, aunque con precisión limitada.
- La red neuronal mejoró significativamente la predicción utilizando todas las variables.
- Los gráficos muestran cómo las predicciones se acercan a los valores reales.
- Se identificaron errores menores (MAE) aceptables para un modelo inicial.

---

## 📁 Archivos principales

📁 vivienda tiempo real/
├── cargar_dataset.py # Carga y visualiza los datos
├── regresion_lineal_simple.py # Implementación de regresión lineal
├── red_neuronal_dos_graficos.py # Red neuronal con visualización
├── README.md # Documento de presentación del proyecto


---

## 🔗 Repositorio

[Ir al repositorio en GitHub](https://github.com/diegope1302/vivienda-red-neuronal)

---

## 👨‍💻 Autor

diego 
Trabajo de Investigación y Desarrollo  
Curso: Ingenieria en Software con Inteligencia Artificial
Modalidad: Individual o pareja


# 🧠 Predicción de Precios de Viviendas con Regresión Lineal y Redes Neuronales

Este proyecto implementa modelos de aprendizaje automático para predecir precios de viviendas utilizando el dataset de California Housing. Se emplean dos enfoques:

1. 📈 **Regresión Lineal Simple**: predicción basada en una sola variable (`AveRooms`).
2. 🤖 **Red Neuronal con TensorFlow/Keras**: modelo de múltiples capas usando todas las variables del dataset.

---

## 📌 Objetivos del proyecto

- Comprender el funcionamiento de una red neuronal artificial.
- Aplicar regresión lineal como método base.
- Usar TensorFlow/Keras para construir y entrenar una red neuronal.
- Visualizar los resultados con Matplotlib.
- Aplicar buenas prácticas de desarrollo como control de versiones con GitHub.

---

## 🗂️ Estructura del proyecto

├── cargar_dataset.py # Carga y visualiza el dataset
├── regresion_lineal_simple.py # Modelo de regresión lineal simple
├── red_neuronal_dos_graficos.py # Red neuronal y visualización por separado
├── README.md # Descripción del proyecto

## ⚙️ Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow

## 📊 Resultados

El modelo de regresión lineal muestra la relación entre AveRooms y el valor medio de vivienda.

La red neuronal mejora la precisión al considerar múltiples variables.

Se generan gráficos que comparan los valores reales con los predichos por ambos modelos.

##📁 Dataset
Se utiliza el dataset California Housing disponible en scikit-learn.

##🔧 Autor
diego
Proyecto de Investigación y Desarrollo – Redes Neuronales
Materia: Inteligencia Artificial / Aprendizaje Automático
Profesor/a: (instructor luis silva)

https://github.com/diegope1302/vivienda-red-neuronal
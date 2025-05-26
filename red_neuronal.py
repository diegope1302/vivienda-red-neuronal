import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Cargar dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear modelo
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X.shape[1]]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compilar modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar
model.fit(X_train, y_train, validation_split=0.2, epochs=50, verbose=0)

# Predecir
y_pred = model.predict(X_test).flatten()

# Crear gráfico combinado lado a lado
plt.figure(figsize=(14, 5))

# Gráfico 1: Valor real
plt.subplot(1, 2, 1)
plt.plot(y_test.values[:100], label="Valor real", color="blue", marker='o')
plt.title("Valores reales")
plt.xlabel("Índice")
plt.ylabel("Precio (en decenas de miles)")
plt.grid(True)
plt.legend()

# Gráfico 2: Valor predicho
plt.subplot(1, 2, 2)
plt.plot(y_pred[:100], label="Valor predicho", color="green", marker='x')
plt.title("Valores predichos")
plt.xlabel("Índice")
plt.ylabel("Precio (en decenas de miles)")
plt.grid(True)
plt.legend()

plt.tight_layout()

# Guardar imagen
plt.savefig("comparacion_valores.png")  # También puedes usar .jpg si prefieres
plt.show()

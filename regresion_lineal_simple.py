import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar el dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Seleccionar una variable (AveRooms) y la variable objetivo (MedHouseVal)
X = df[["AveRooms"]]  # Variable independiente
y = df["MedHouseVal"] # Variable dependiente

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Mostrar resultados
plt.scatter(X_test, y_test, color="blue", label="Datos reales", alpha=0.5)
plt.plot(X_test, y_pred, color="red", label="Predicción", linewidth=2)
plt.xlabel("Promedio de habitaciones (AveRooms)")
plt.ylabel("Valor medio de vivienda (MedHouseVal)")
plt.title("Regresión Lineal Simple")
plt.legend()
plt.grid(True)
plt.show()

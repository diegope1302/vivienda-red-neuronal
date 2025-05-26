from sklearn.datasets import fetch_california_housing
import pandas as pd

# Cargar dataset California housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Mostrar las primeras filas
print(df.head())

# Mostrar las columnas disponibles
print("Columnas disponibles:", df.columns.tolist())

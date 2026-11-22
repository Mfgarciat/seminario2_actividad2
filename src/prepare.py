from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print("=== Etapa: preparación de datos ===")

# Cargar datos de Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["species"] = iris.target

# Dividir en train/test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Guardar
ruta_train = Path("data/processed/train.csv")
ruta_test = Path("data/processed/test.csv")

train.to_csv(ruta_train, index=False)
test.to_csv(ruta_test, index=False)

print(f"Dataset Iris cargado: {len(data)} filas")
print(f"Train: {len(train)} filas -> {ruta_train}")
print(f"Test:  {len(test)} filas -> {ruta_test}")
print("Etapa de preparación finalizada.")

import csv
from pathlib import Path

ruta = Path("data/processed/train.csv")

if not ruta.exists():
    print(f"No existe {ruta}")
    print("Ejecuta primero: python src/prepare.py")
    raise SystemExit(1)

# leer el csv
with open(ruta, encoding="utf-8", newline="") as f:
    lector = csv.DictReader(f)
    filas = list(lector)
    columnas = lector.fieldnames

if not filas:
    print("train.csv esta vacio")
    raise SystemExit(1)

# corromper la primera fila:
# - un valor fuera de rango
# - un nulo
filas[0]["sepal length (cm)"] = "99.0"
filas[0]["sepal width (cm)"] = ""

# guardar de nuevo
with open(ruta, "w", encoding="utf-8", newline="") as f:
    escritor = csv.DictWriter(f, fieldnames=columnas)
    escritor.writeheader()
    escritor.writerows(filas)

print(f"Datos corruptos en {ruta}")
print("  - sepal length (cm) = 99.0 (fuera de rango)")
print("  - sepal width (cm) = '' (nulo)")
print("Ahora ejecuta: python validate_data.py")

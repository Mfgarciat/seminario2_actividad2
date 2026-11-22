import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# etapas del pipeline en orden
etapas = [
    ("Preparacion de datos", [sys.executable, "src/prepare.py"]),
    ("Validacion de datos", [sys.executable, "validate_data.py"]),
    ("Entrenamiento del modelo", [sys.executable, "src/train.py"]),
    ("Pruebas automatizadas", [sys.executable, "-m", "pytest", "tests/", "-v"]),
]

def correr_etapa(nombre, comando):
    print()
    print("=" * 55)
    print(f"Iniciando etapa: {nombre}")
    print("=" * 55)

    inicio = time.perf_counter()
    resultado = subprocess.run(comando, cwd=ROOT)
    tiempo = time.perf_counter() - inicio

    # si falla, detenemos todo
    if resultado.returncode != 0:
        print()
        print("-" * 55)
        print(f"PIPELINE DETENIDO: fallo la etapa '{nombre}'")
        print("-" * 55)
        sys.exit(resultado.returncode)

    print(f"Etapa completada en {tiempo:.2f} segundos.")


print("=" * 55)
print("INICIANDO PIPELINE IRIS")
print("=" * 55)

for nombre, comando in etapas:
    correr_etapa(nombre, comando)

print()
print("-" * 43)
print("PIPELINE FINALIZADO CORRECTAMENTE")
print("-" * 43)
print("Los datos fueron preparados y validados.")
print("El modelo fue entrenado y almacenado.")
print("Las pruebas automatizadas fueron superadas.")

import csv
import json
import sys
from pathlib import Path


def cargar_contrato():
    """Carga el contrato de datos."""
    with open("data_contract.json", encoding="utf-8") as f:
        return json.load(f)


def armar_reglas(contrato, incluir_target=True):
    """Saca las reglas del contrato para poder validar."""
    reglas = {}

    for campo in contrato["entrada_inferencia"]["campos"]:
        reglas[campo["nombre"]] = campo

    # species solo aplica en train/test, no en inferencia
    if incluir_target:
        target = contrato["variable_objetivo"]
        reglas[target["nombre"]] = target

    return reglas


def es_nulo(valor):
    # Consideramos nulo: None, vacio, "null", "none", "nan"
    if valor is None:
        return True
    texto = str(valor).strip().lower()
    if texto == "" or texto in ["null", "none", "nan"]:
        return True
    return False


def validar_fila(fila, reglas, archivo, num_fila):
    """Valida una fila y devuelve lista de errores."""
    errores = []

    for nombre, regla in reglas.items():
        # 1) campo requerido
        if nombre not in fila:
            if regla.get("required", True):
                errores.append(f"{archivo} fila {num_fila}: falta la columna requerida '{nombre}'")
            continue

        valor = fila[nombre]

        # 2) nulos
        if es_nulo(valor):
            if not regla.get("nullable", True):
                errores.append(f"{archivo} fila {num_fila}: '{nombre}' no puede ser nulo")
            continue

        tipo = regla.get("tipo")

        # 3) tipos + rangos
        if tipo == "float":
            try:
                numero = float(valor)
            except (TypeError, ValueError):
                errores.append(f"{archivo} fila {num_fila}: '{nombre}' debe ser float, recibido '{valor}'")
                continue

            # rangos min/max
            if "min" in regla and numero < regla["min"]:
                errores.append(
                    f"{archivo} fila {num_fila}: '{nombre}'={numero} esta por debajo del minimo {regla['min']}"
                )
            if "max" in regla and numero > regla["max"]:
                errores.append(
                    f"{archivo} fila {num_fila}: '{nombre}'={numero} esta por encima del maximo {regla['max']}"
                )

        elif tipo == "int":
            try:
                numero = float(valor)
                if not numero.is_integer():
                    raise ValueError
                numero = int(numero)
            except (TypeError, ValueError):
                errores.append(f"{archivo} fila {num_fila}: '{nombre}' debe ser int, recibido '{valor}'")
                continue

            # valores permitidos (para species)
            permitidos = regla.get("valores_permitidos")
            if permitidos is not None and numero not in permitidos:
                errores.append(
                    f"{archivo} fila {num_fila}: '{nombre}'={numero} no esta en {permitidos}"
                )

    return errores


def validate_inference_example(ejemplo, contrato=None):
    """Valida un ejemplo de inferencia (sin species)."""
    if contrato is None:
        contrato = cargar_contrato()

    reglas = armar_reglas(contrato, incluir_target=False)

    # quitamos motivos_invalidez si viene en el ejemplo del contrato
    limpio = {}
    for k, v in ejemplo.items():
        if k != "motivos_invalidez":
            limpio[k] = v

    return validar_fila(limpio, reglas, "contrato", 1)


def validar_archivo(ruta, reglas):
    """Valida un csv completo."""
    path = Path(ruta)
    if not path.exists():
        return [f"No existe el archivo {ruta}"]

    errores = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # columnas requeridas
        esperadas = []
        for nombre, regla in reglas.items():
            if regla.get("required", True):
                esperadas.append(nombre)

        columnas = reader.fieldnames or []
        faltantes = []
        for col in esperadas:
            if col not in columnas:
                faltantes.append(col)

        if faltantes:
            errores.append(f"{ruta}: faltan columnas requeridas {faltantes}")
            return errores

        # validar cada fila (empezamos en 2 por el header)
        num_fila = 2
        for fila in reader:
            errores.extend(validar_fila(fila, reglas, ruta, num_fila))
            num_fila += 1

    return errores


def main():
    print("=== Etapa: validacion de datos ===")

    if not Path("data_contract.json").exists():
        print("ERROR: no se encontro data_contract.json")
        return 1

    contrato = cargar_contrato()
    reglas = armar_reglas(contrato)
    print(f"Contrato cargado ({len(reglas)} reglas)")

    archivos = [
        "data/processed/train.csv",
        "data/processed/test.csv",
    ]

    todos_los_errores = []
    for archivo in archivos:
        todos_los_errores.extend(validar_archivo(archivo, reglas))

    if todos_los_errores:
        print("Validacion de datos FALLIDA:")
        for error in todos_los_errores:
            print(f"  - {error}")
        return 1

    print("Validacion de datos OK:")
    for archivo in archivos:
        print(f"  - {archivo}")
    print("Etapa de validacion finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

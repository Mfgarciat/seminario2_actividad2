# Actividad 2 - Pipeline Iris y calidad de datos

Pipeline para entrenar un Random Forest con el dataset Iris, validar los datos con un contrato JSON y registrar el experimento en MLflow.

## Requisitos

- Python 3.10 o superior

## Estructura

```text
actividad2_seminario/
├── run_pipeline.py      # corre todo el flujo
├── validate_data.py     # valida tipos, nulos, rangos y required
├── data_contract.json   # contrato de entrada
├── romper_datos.py      # corrompe datos para probar la validacion
├── ejemplos/
│   ├── entrada_valida.json
│   └── entrada_invalida.json
├── requirements.txt
├── readme.md
├── src/
│   ├── prepare.py       # genera train.csv y test.csv
│   └── train.py         # entrena el modelo + MLflow
├── tests/
│   └── test_model.py
├── data/processed/
└── models/
```

## Como ejecutarlo

```bash
cd actividad2_seminario
python -m venv venv
source venv/bin/activate   # en Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

Eso corre en orden:

1. `src/prepare.py` - prepara los datos
2. `validate_data.py` - valida segun el contrato
3. `src/train.py` - entrena y guarda `models/model.pkl`
4. `pytest tests/` - pruebas del modelo y del contrato

Si la validacion falla, el pipeline se detiene.

Tambien se puede correr por partes:

```bash
python src/prepare.py
python validate_data.py
python src/train.py
pytest tests/
```

## Contrato de datos

`data_contract.json` define las reglas de entrada:

- campos requeridos
- tipo (float / int)
- no nulos
- rangos min/max
- `species` con valores 0, 1, 2 (solo en train/test)

Tambien trae un `ejemplo_valido` y un `ejemplo_invalido`.

| Campo | Tipo | Nulo | Rango |
|-------|------|------|-------|
| sepal length (cm) | float | No | 4.0 - 8.0 |
| sepal width (cm) | float | No | 2.0 - 4.5 |
| petal length (cm) | float | No | 1.0 - 7.0 |
| petal width (cm) | float | No | 0.1 - 2.5 |
| species | int | No | 0, 1, 2 |

## Ejemplos de inferencia (valido / invalido)

```bash
python validate_data.py --input ejemplos/entrada_valida.json
python validate_data.py --input ejemplos/entrada_invalida.json
```

- La valida debe pasar.
- La invalida debe fallar por 4 motivos: 9.5 fuera de rango, sepal width nulo, petal length no numerico y -1.0 bajo el minimo.

## Probar que la validacion falla (romper CSV)

```bash
python src/prepare.py
python romper_datos.py
python validate_data.py
```

Deberia fallar por valor fuera de rango y nulo. Para volver a datos limpios:

```bash
python run_pipeline.py
```

## MLflow (opcional)

```bash
mlflow ui
```

Luego abrir http://127.0.0.1:5000

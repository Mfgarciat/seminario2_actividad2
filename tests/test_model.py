import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from validate_data import cargar_contrato, validate_inference_example


def test_data_loading():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    assert not train.empty and not test.empty
    assert "species" in train.columns


def test_model_exists():
    model = joblib.load("models/model.pkl")
    assert model is not None


def test_model_can_be_loaded():
    model = joblib.load("models/model.pkl")
    assert hasattr(model, "predict")


def test_model_accuracy():
    model = joblib.load("models/model.pkl")
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop("species", axis=1)
    y_test = test["species"]

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.8


def test_contract_valid_example_passes():
    contract = cargar_contrato()
    errors = validate_inference_example(contract["ejemplo_valido"], contract)
    assert errors == [], f"El ejemplo válido no debió fallar: {errors}"


def test_contract_invalid_example_is_rejected():
    contract = cargar_contrato()
    errors = validate_inference_example(contract["ejemplo_invalido"], contract)
    assert errors, "El ejemplo inválido debió producir errores de validación"

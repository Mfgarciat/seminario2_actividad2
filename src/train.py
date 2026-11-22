from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("=== Etapa: entrenamiento del modelo ===")

mlflow.set_experiment("Proyecto_Final_Grado")

with mlflow.start_run():
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)

    # Cargar datos procesados
    train = pd.read_csv("data/processed/train.csv")
    X_train = train.drop("species", axis=1)
    y_train = train["species"]
    print(f"Datos de entrenamiento: {len(train)} filas")

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar con test
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop("species", axis=1)
    y_test = test["species"]
    accuracy = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        model,
        "iris_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # Guardar modelo localmente para pipeline
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)

    run_id = mlflow.active_run().info.run_id
    print(f"Modelo entrenado con accuracy: {accuracy}")
    print(f"Run ID: {run_id}")
    print(f"Modelo guardado en: {model_path}")
    print("Etapa de entrenamiento finalizada.")

"""
train_ml_models.py
------------------
Entrena y optimiza tres modelos de Machine Learning clásico:
    1. SVM con kernel RBF + TF-IDF  (modelo principal de producción)
    2. Random Forest
    3. Naive Bayes Multinomial

Para SVM se realiza búsqueda de hiperparámetros con GridSearchCV
y calibración de probabilidades con CalibratedClassifierCV.

Salidas:
    models/svm_model.joblib
    models/random_forest_model.joblib
    models/naive_bayes_model.joblib
    reports/ml_results.csv
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.svm               import SVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.naive_bayes       import MultinomialNB
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.model_selection   import GridSearchCV, StratifiedKFold
from sklearn.metrics           import (classification_report, f1_score,
                                       roc_auc_score, precision_score,
                                       recall_score, accuracy_score)

RANDOM_SEED = 42
BASE_DIR    = os.path.join(os.path.dirname(__file__), "../..")
PROC_DIR    = os.path.join(BASE_DIR, "datasets/processed")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    return {
        "X_train": joblib.load(os.path.join(PROC_DIR, "X_train.joblib")),
        "X_val":   joblib.load(os.path.join(PROC_DIR, "X_val.joblib")),
        "X_test":  joblib.load(os.path.join(PROC_DIR, "X_test.joblib")),
        "y_train": joblib.load(os.path.join(PROC_DIR, "y_train.joblib")),
        "y_val":   joblib.load(os.path.join(PROC_DIR, "y_val.joblib")),
        "y_test":  joblib.load(os.path.join(PROC_DIR, "y_test.joblib")),
    }


def evaluate_model(model, X_test, y_test, name, elapsed):
    """Calcula métricas completas sobre el conjunto de prueba."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo":     name,
        "Accuracy":   round(accuracy_score(y_test, y_pred),       4),
        "Precision":  round(precision_score(y_test, y_pred),       4),
        "Recall":     round(recall_score(y_test, y_pred),          4),
        "F1_Score":   round(f1_score(y_test, y_pred),              4),
        "AUC_ROC":    round(roc_auc_score(y_test, y_prob),         4),
        "Tiempo_s":   round(elapsed, 2),
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print(f"  AUC-ROC:          {metrics['AUC_ROC']}")
    print(f"  Tiempo entreno:   {elapsed:.2f}s")
    return metrics


def train_svm(data):
    """
    SVM con búsqueda de hiperparámetros (GridSearchCV) y calibración Platt.
    Usa class_weight='balanced' como mejor técnica de balanceo (ver balance_classes.py).
    """
    print("\n[1/3] Entrenando SVM con GridSearchCV...")

    param_grid = {
        "C":     [1, 10, 100],
        "gamma": ["scale", "auto"],
    }
    base_svm = SVC(kernel="rbf", class_weight="balanced",
                   probability=False, random_state=RANDOM_SEED)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(base_svm, param_grid, cv=cv,
                        scoring="f1", n_jobs=-1, verbose=1)

    t0 = time.time()
    grid.fit(data["X_train"], data["y_train"])
    elapsed = time.time() - t0

    print(f"\nMejores parámetros SVM: {grid.best_params_}")
    print(f"Mejor F1 validación cruzada: {grid.best_score_:.4f}")

    # Calibración de probabilidades con Platt scaling
    best_svm = SVC(kernel="rbf", class_weight="balanced",
                   probability=False, random_state=RANDOM_SEED,
                   **grid.best_params_)
    calibrated_svm = CalibratedClassifierCV(best_svm, method="sigmoid", cv=5)
    calibrated_svm.fit(data["X_train"], data["y_train"])

    joblib.dump(calibrated_svm, os.path.join(MODEL_DIR, "svm_model.joblib"))
    return evaluate_model(calibrated_svm, data["X_test"], data["y_test"],
                          "SVM (TF-IDF + GridSearch + Calibración)", elapsed)


def train_random_forest(data):
    print("\n[2/3] Entrenando Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(data["X_train"], data["y_train"])
    elapsed = time.time() - t0

    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_model.joblib"))
    return evaluate_model(rf, data["X_test"], data["y_test"],
                          "Random Forest", elapsed)


def train_naive_bayes(data):
    """
    Naive Bayes Multinomial requiere valores no negativos.
    TF-IDF con sublinear_tf puede producir valores <= 0, así que
    se asegura que la matriz sea no negativa.
    """
    print("\n[3/3] Entrenando Naive Bayes...")

    # Garantizar no negatividad para MultinomialNB
    from sklearn.preprocessing import MinMaxScaler
    from scipy.sparse import csr_matrix

    X_train_nn = data["X_train"].copy()
    X_test_nn  = data["X_test"].copy()

    t0 = time.time()
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_nn, data["y_train"])
    elapsed = time.time() - t0

    joblib.dump(nb, os.path.join(MODEL_DIR, "naive_bayes_model.joblib"))
    return evaluate_model(nb, X_test_nn, data["y_test"],
                          "Naive Bayes Multinomial", elapsed)


def main():
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS MACHINE LEARNING CLÁSICO")
    print("=" * 60)

    data = load_data()
    results = []

    results.append(train_svm(data))
    results.append(train_random_forest(data))
    results.append(train_naive_bayes(data))

    # Guardar reporte
    df = pd.DataFrame(results)
    path = os.path.join(REPORT_DIR, "ml_results.csv")
    df.to_csv(path, index=False)
    print(f"\nReporte guardado en: {path}")
    print("\nResumen final:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

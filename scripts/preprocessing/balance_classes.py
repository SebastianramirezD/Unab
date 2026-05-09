"""
balance_classes.py
------------------
Implementa y compara tres técnicas de balanceo de clases sobre el conjunto
de entrenamiento:

    1. SMOTE       — Oversampling sintético (imblearn)
    2. Random Undersampling — Submuestreo aleatorio de la clase mayoritaria
    3. Class Weight Adjustment — Sin resampling, ajuste de pesos en el modelo

Genera un reporte comparativo en reports/balanceo_comparativo.csv
"""

import os
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler

RANDOM_SEED = 42
BASE_DIR   = os.path.join(os.path.dirname(__file__), "../..")
PROC_DIR   = os.path.join(BASE_DIR, "datasets/processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    X_train = joblib.load(os.path.join(PROC_DIR, "X_train.joblib"))
    X_val   = joblib.load(os.path.join(PROC_DIR, "X_val.joblib"))
    y_train = joblib.load(os.path.join(PROC_DIR, "y_train.joblib"))
    y_val   = joblib.load(os.path.join(PROC_DIR, "y_val.joblib"))
    return X_train, X_val, y_train, y_val


def train_and_evaluate(X_tr, y_tr, X_val, y_val, class_weight=None, label=""):
    """Entrena SVM y retorna métricas sobre validación."""
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              class_weight=class_weight, random_state=RANDOM_SEED)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)

    f1_spam    = f1_score(y_val, y_pred, pos_label=1)
    f1_ham     = f1_score(y_val, y_pred, pos_label=0)
    f1_macro   = f1_score(y_val, y_pred, average="macro")

    print(f"\n{'─'*50}")
    print(f"Técnica: {label}")
    print(f"Distribución entrenamiento: {Counter(y_tr)}")
    print(classification_report(y_val, y_pred, target_names=["ham", "spam"]))

    return {
        "Tecnica":       label,
        "F1_spam":       round(f1_spam,  4),
        "F1_ham":        round(f1_ham,   4),
        "F1_macro":      round(f1_macro, 4),
        "n_train_spam":  int(Counter(y_tr)[1]),
        "n_train_ham":   int(Counter(y_tr)[0]),
    }


def main():
    print("=" * 60)
    print("COMPARACIÓN DE TÉCNICAS DE BALANCEO DE CLASES")
    print("=" * 60)

    X_train, X_val, y_train, y_val = load_data()
    print(f"\nDistribución original: {Counter(y_train)}")

    results = []

    # ── 1. Sin balanceo (baseline) ────────────────────────────────────────────
    res = train_and_evaluate(X_train, y_train, X_val, y_val,
                             class_weight=None,
                             label="Sin balanceo (baseline)")
    results.append(res)

    # ── 2. SMOTE (oversampling) ───────────────────────────────────────────────
    smote = SMOTE(random_state=RANDOM_SEED)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    res = train_and_evaluate(X_smote, y_smote, X_val, y_val,
                             class_weight=None,
                             label="Oversampling (SMOTE)")
    results.append(res)

    # ── 3. Random Undersampling ───────────────────────────────────────────────
    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_under, y_under = rus.fit_resample(X_train, y_train)
    res = train_and_evaluate(X_under, y_under, X_val, y_val,
                             class_weight=None,
                             label="Undersampling (Random)")
    results.append(res)

    # ── 4. Class Weight Adjustment ────────────────────────────────────────────
    res = train_and_evaluate(X_train, y_train, X_val, y_val,
                             class_weight="balanced",
                             label="Class Weight Adjustment")
    results.append(res)

    # ── Reporte CSV ───────────────────────────────────────────────────────────
    df_report = pd.DataFrame(results)
    report_path = os.path.join(REPORT_DIR, "balanceo_comparativo.csv")
    df_report.to_csv(report_path, index=False)
    print(f"\nReporte guardado en: {report_path}")
    print("\nResumen:")
    print(df_report.to_string(index=False))

    # Guardar el mejor dataset balanceado (class weight no genera dataset nuevo)
    print("\nGuardando datos SMOTE para referencia...")
    joblib.dump(X_smote, os.path.join(PROC_DIR, "X_train_smote.joblib"))
    joblib.dump(y_smote, os.path.join(PROC_DIR, "y_train_smote.joblib"))
    print("Completado.")


if __name__ == "__main__":
    main()

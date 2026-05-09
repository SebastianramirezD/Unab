"""
shap_analysis.py
----------------
Análisis de interpretabilidad con SHAP (SHapley Additive exPlanations)
para el modelo SVM de producción.

Genera:
    reports/shap_summary.png      — Importancia global de características
    reports/shap_bar.png          — Top 20 características más influyentes
    reports/shap_examples.json    — Explicaciones para ejemplos individuales
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # backend sin pantalla

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR   = os.path.join(os.path.dirname(__file__), "../..")
PROC_DIR   = os.path.join(BASE_DIR, "datasets/processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_artifacts():
    """Carga el modelo SVM, vectorizador y datos de prueba."""
    model      = joblib.load(os.path.join(MODEL_DIR,  "svm_model.joblib"))
    vectorizer = joblib.load(os.path.join(PROC_DIR,   "vectorizer.joblib"))
    X_test     = joblib.load(os.path.join(PROC_DIR,   "X_test.joblib"))
    y_test     = joblib.load(os.path.join(PROC_DIR,   "y_test.joblib"))
    X_test_raw = joblib.load(os.path.join(PROC_DIR,   "X_test_raw.joblib"))
    return model, vectorizer, X_test, y_test, X_test_raw


def explain_global(model, X_test, feature_names):
    """
    Genera explicaciones SHAP globales usando LinearExplainer.
    Para CalibratedClassifierCV extraemos el estimador base.
    """
    print("Generando SHAP values globales (puede tardar ~1 min)...")

    # Para CalibratedClassifierCV necesitamos el estimador calibrado
    # Usamos KernelExplainer con una muestra reducida para eficiencia
    background = shap.sample(X_test, 100, random_state=RANDOM_SEED)

    def predict_proba_spam(x):
        return model.predict_proba(x)[:, 1]

    explainer  = shap.KernelExplainer(predict_proba_spam, background)
    sample_idx = np.random.choice(X_test.shape[0], 200, replace=False)
    X_sample   = X_test[sample_idx]

    shap_values = explainer.shap_values(X_sample, nsamples=100)

    # ── Gráfico de resumen (beeswarm) ────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names,
                      max_display=20,
                      show=False,
                      plot_type="dot")
    plt.title("SHAP Summary Plot — Top 20 características más influyentes",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary guardado en: {path}")

    # ── Gráfico de barras (importancia media |SHAP|) ─────────────────────────
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[-20:][::-1]
    top_feats = [feature_names[i] for i in top_idx]
    top_vals  = mean_shap[top_idx]

    plt.figure(figsize=(10, 7))
    colors = ["#d32f2f" if v > 0 else "#1976d2" for v in top_vals]
    plt.barh(range(len(top_feats)), top_vals, color="#2196F3", alpha=0.85)
    plt.yticks(range(len(top_feats)), top_feats, fontsize=10)
    plt.xlabel("Importancia media |SHAP value|", fontsize=11)
    plt.title("Top 20 Características — Importancia SHAP Global",
              fontsize=12, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar plot guardado en: {path}")

    return explainer, shap_values, sample_idx


def explain_individual(model, explainer, X_test, X_test_raw, y_test,
                        feature_names, n_examples=5):
    """
    Genera explicaciones SHAP para N ejemplos individuales:
    - 3 spam correctamente clasificados
    - 2 falsos negativos (spam clasificado como ham)
    """
    print(f"\nGenerando explicaciones para {n_examples} ejemplos individuales...")

    y_pred = model.predict(X_test)

    # Índices de spam correcto (TP) y falsos negativos (FN)
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    examples = []
    for idx in tp_idx[:3]:
        sv = explainer.shap_values(X_test[idx], nsamples=100)
        top = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:10]
        examples.append({
            "tipo":         "Verdadero Positivo (spam detectado correctamente)",
            "texto":        X_test_raw[idx],
            "prediccion":   "spam",
            "real":         "spam",
            "top_features": [{"feature": f, "shap_value": round(float(v), 5)} for f, v in top]
        })

    for idx in fn_idx[:2]:
        sv = explainer.shap_values(X_test[idx], nsamples=100)
        top = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:10]
        examples.append({
            "tipo":         "Falso Negativo (spam no detectado)",
            "texto":        X_test_raw[idx],
            "prediccion":   "ham",
            "real":         "spam",
            "top_features": [{"feature": f, "shap_value": round(float(v), 5)} for f, v in top]
        })

    path = os.path.join(REPORT_DIR, "shap_examples.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Ejemplos SHAP guardados en: {path}")

    # Mostrar resumen en consola
    for ex in examples:
        print(f"\n{'─'*55}")
        print(f"Tipo: {ex['tipo']}")
        print(f"Texto: {ex['texto'][:80]}...")
        print("Top características:")
        for feat in ex["top_features"][:5]:
            direction = "→ spam" if feat["shap_value"] > 0 else "→ ham"
            print(f"  {feat['feature']:20s}  SHAP={feat['shap_value']:+.4f}  {direction}")


def main():
    print("=" * 60)
    print("ANÁLISIS DE INTERPRETABILIDAD CON SHAP")
    print("=" * 60)

    model, vectorizer, X_test, y_test, X_test_raw = load_artifacts()
    feature_names = vectorizer.get_feature_names_out().tolist()

    print(f"Modelo cargado: {type(model).__name__}")
    print(f"Características TF-IDF: {len(feature_names)}")
    print(f"Ejemplos de prueba: {X_test.shape[0]}")

    explainer, shap_values, sample_idx = explain_global(model, X_test, feature_names)

    # Usar los índices de muestra para alinear X_test_raw
    X_test_raw_sample = X_test_raw[sample_idx]
    y_test_sample     = y_test[sample_idx]

    explain_individual(model, explainer, X_test, X_test_raw,
                       y_test, feature_names, n_examples=5)

    print("\nAnálisis SHAP completado exitosamente.")


if __name__ == "__main__":
    main()

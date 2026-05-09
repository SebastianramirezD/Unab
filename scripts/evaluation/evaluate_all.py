"""
evaluate_all.py
---------------
Evaluación comparativa de todos los modelos entrenados (ML + DL)
sobre el conjunto de prueba. Genera:
    - reports/comparacion_final.csv
    - reports/confusion_matrices.png
    - reports/roc_curves.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score, roc_auc_score,
    precision_score, recall_score, accuracy_score
)

RANDOM_SEED = 42
BASE_DIR    = os.path.join(os.path.dirname(__file__), "../..")
PROC_DIR    = os.path.join(BASE_DIR, "datasets/processed")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_test_data():
    X_test = joblib.load(os.path.join(PROC_DIR, "X_test.joblib"))
    y_test = joblib.load(os.path.join(PROC_DIR, "y_test.joblib"))
    return X_test, y_test


def evaluate_sklearn_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Modelo":    name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1_Score":  round(f1_score(y_test, y_pred),        4),
        "AUC_ROC":   round(roc_auc_score(y_test, y_prob),   4),
    }, y_pred, y_prob


def plot_confusion_matrices(cms, names):
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4))
    if len(cms) == 1:
        axes = [axes]
    for ax, cm, name in zip(axes, cms, names):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["ham", "spam"],
                    yticklabels=["ham", "spam"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicho")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Matrices de confusión guardadas en: {path}")


def plot_roc_curves(roc_data):
    plt.figure(figsize=(8, 6))
    for name, fpr, tpr, auc_score in roc_data:
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Tasa de Falsos Positivos", fontsize=12)
    plt.ylabel("Tasa de Verdaderos Positivos", fontsize=12)
    plt.title("Curvas ROC — Comparación de Modelos", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    path = os.path.join(REPORT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curvas ROC guardadas en: {path}")


def plot_metrics_comparison(df):
    metrics = ["Precision", "Recall", "F1_Score", "AUC_ROC"]
    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df[metric], width, label=metric, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Modelo", fontsize=12)
    ax.set_ylabel("Valor de la métrica", fontsize=12)
    ax.set_title("Comparación de Métricas por Modelo", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Modelo"], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de comparación guardado en: {path}")


def main():
    print("=" * 60)
    print("EVALUACIÓN COMPARATIVA DE TODOS LOS MODELOS")
    print("=" * 60)

    X_test, y_test = load_test_data()
    results, cms, roc_data = [], [], []

    models_ml = [
        ("svm_model.joblib",           "SVM (TF-IDF)"),
        ("random_forest_model.joblib", "Random Forest"),
        ("naive_bayes_model.joblib",   "Naive Bayes"),
    ]

    for fname, name in models_ml:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"ADVERTENCIA: {fname} no encontrado. Ejecuta train_ml_models.py primero.")
            continue
        model = joblib.load(path)
        metrics, y_pred, y_prob = evaluate_sklearn_model(model, X_test, y_test, name)
        results.append(metrics)
        cms.append(confusion_matrix(y_test, y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((name, fpr, tpr, metrics["AUC_ROC"]))
        print(f"\n{name}:")
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    if results:
        df = pd.DataFrame(results)
        plot_confusion_matrices(cms, [r["Modelo"] for r in results])
        plot_roc_curves(roc_data)
        plot_metrics_comparison(df)

        # Combinar con resultados DL si existen
        dl_path = os.path.join(REPORT_DIR, "dl_results.csv")
        if os.path.exists(dl_path):
            df_dl = pd.read_csv(dl_path)[["Modelo","Accuracy","Precision","Recall","F1_Score","AUC_ROC"]]
            df = pd.concat([df, df_dl], ignore_index=True)

        final_path = os.path.join(REPORT_DIR, "comparacion_final.csv")
        df.to_csv(final_path, index=False)
        print(f"\nReporte final guardado en: {final_path}")
        print("\nRESUMEN COMPARATIVO FINAL:")
        print(df.sort_values("F1_Score", ascending=False).to_string(index=False))
    else:
        print("\nNo se encontraron modelos entrenados. Ejecuta los scripts de entrenamiento primero.")


if __name__ == "__main__":
    main()

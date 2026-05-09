# notebooks/01_eda.py
# ============================================================
# Análisis Exploratorio de Datos (EDA)
# SMS Spam Collection Dataset
# ACIF104 — Aprendizaje de Máquinas — Fase 3
# ============================================================
# Este archivo puede ejecutarse como script Python o abrirse
# en Jupyter / VS Code como notebook (con extensión Jupytext).

# %% [markdown]
# ## 1. Importaciones y configuración

# %%
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid", palette="muted")

RANDOM_SEED = 42
DATA_PATH = "../datasets/raw/spam.csv"
REPORT_DIR = "../reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# %% [markdown]
# ## 2. Carga de datos

# %%
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"\nPrimeras filas:")
df.head(10)

# %% [markdown]
# ## 3. Distribución de clases

# %%
counts = df["label"].value_counts()
print("Distribución de clases:")
print(counts)
print(f"\nDesbalance: {counts['ham'] / counts['spam']:.1f}x más ham que spam")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico de barras
axes[0].bar(counts.index, counts.values, color=["#42A5F5", "#EF5350"], edgecolor="white", linewidth=1.5)
axes[0].set_title("Distribución de clases", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Cantidad de mensajes")
for i, (label, count) in enumerate(zip(counts.index, counts.values)):
    axes[0].text(i, count + 30, f"{count}\n({count/len(df)*100:.1f}%)",
                 ha="center", fontsize=11, fontweight="bold")

# Torta
axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#42A5F5", "#EF5350"], startangle=90,
            textprops={"fontsize": 13})
axes[1].set_title("Proporción de clases", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/eda_class_distribution.png", bbox_inches="tight")
plt.show()
print(f"Guardado: eda_class_distribution.png")

# %% [markdown]
# ## 4. Análisis de longitud de mensajes

# %%
df["msg_length"] = df["text"].apply(len)
df["word_count"] = df["text"].apply(lambda x: len(x.split()))

stats = df.groupby("label")[["msg_length", "word_count"]].describe()
print("Estadísticas de longitud por clase:")
print(stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in zip(axes,
                           ["msg_length", "word_count"],
                           ["Longitud del mensaje (caracteres)", "Número de palabras"]):
    for label, color in zip(["ham", "spam"], ["#42A5F5", "#EF5350"]):
        subset = df[df["label"] == label][col]
        ax.hist(subset, bins=40, alpha=0.65, label=label, color=color, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(title)
    ax.set_ylabel("Frecuencia")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/eda_message_length.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Características estructurales del texto

# %%
df["upper_ratio"]   = df["text"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
df["url_count"]     = df["text"].apply(lambda x: len(re.findall(r"http\S+|www\S+", x)))
df["exclaim_count"] = df["text"].apply(lambda x: x.count("!"))
df["num_count"]     = df["text"].apply(lambda x: len(re.findall(r"\d+", x)))

feat_cols = ["upper_ratio", "url_count", "exclaim_count", "num_count"]
print("\nPromedios por clase:")
print(df.groupby("label")[feat_cols].mean().round(3))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, feat_cols):
    for label, color in zip(["ham", "spam"], ["#42A5F5", "#EF5350"]):
        subset = df[df["label"] == label][col]
        ax.hist(subset, bins=25, alpha=0.65, label=label, color=color, edgecolor="white")
    ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/eda_structural_features.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Palabras más frecuentes por clase

# %%
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP = set(stopwords.words("english"))

def get_top_words(texts, n=20):
    words = []
    for t in texts:
        for w in re.sub(r"[^a-zA-Z\s]", "", t.lower()).split():
            if w not in STOP and len(w) > 2:
                words.append(w)
    return Counter(words).most_common(n)

ham_words  = get_top_words(df[df["label"] == "ham"]["text"])
spam_words = get_top_words(df[df["label"] == "spam"]["text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, words, title, color in zip(
    axes,
    [ham_words, spam_words],
    ["Top 20 palabras — HAM", "Top 20 palabras — SPAM"],
    ["#42A5F5", "#EF5350"]
):
    df_w = pd.DataFrame(words, columns=["word", "count"])
    ax.barh(df_w["word"][::-1], df_w["count"][::-1], color=color, alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Frecuencia")

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/eda_top_words.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. WordClouds por clase

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, label, color in zip(axes, ["ham", "spam"], ["Blues", "Reds"]):
    text = " ".join(df[df["label"] == label]["text"].values)
    wc = WordCloud(width=600, height=300, background_color="white",
                   colormap=color, max_words=100,
                   stopwords=STOP).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud — {label.upper()}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/eda_wordclouds.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Correlación de características

# %%
num_feats = ["msg_length", "word_count", "upper_ratio", "url_count", "exclaim_count", "num_count"]
df["is_spam"] = (df["label"] == "spam").astype(int)

corr = df[num_feats + ["is_spam"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5)
plt.title("Matriz de correlación de características", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/eda_correlation.png", bbox_inches="tight")
plt.show()

print("\n✅ EDA completado. Reportes guardados en:", REPORT_DIR)

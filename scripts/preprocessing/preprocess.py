"""
preprocess.py
-------------
Limpieza y vectorización del dataset SMS Spam Collection.
Genera los conjuntos de entrenamiento, validación y prueba (70/15/15)
con estratificación para preservar la proporción de clases.

Salidas:
    datasets/processed/X_train.joblib
    datasets/processed/X_val.joblib
    datasets/processed/X_test.joblib
    datasets/processed/y_train.joblib
    datasets/processed/y_val.joblib
    datasets/processed/y_test.joblib
    datasets/processed/vectorizer.joblib
"""

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Garantizar reproducibilidad en todos los splits
RANDOM_SEED = 42

# Rutas
BASE_DIR  = os.path.join(os.path.dirname(__file__), "../..")
RAW_CSV   = os.path.join(BASE_DIR, "datasets/raw/spam.csv")
PROC_DIR  = os.path.join(BASE_DIR, "datasets/processed")

os.makedirs(PROC_DIR, exist_ok=True)

# Descargar recursos NLTK necesarios (solo la primera vez)
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


# ─── Funciones de limpieza ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Aplica un pipeline de limpieza de texto:
    1. Minúsculas
    2. Elimina URLs
    3. Elimina números
    4. Elimina puntuación
    5. Elimina stopwords
    6. Normaliza espacios
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)          # reemplazar URLs
    text = re.sub(r"\d+", " num ", text)                       # reemplazar números
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características estructurales al DataFrame:
    - Longitud del mensaje
    - Proporción de mayúsculas
    - Conteo de URLs
    - Conteo de signos de exclamación
    """
    df = df.copy()
    df["msg_length"]    = df["text"].apply(len)
    df["upper_ratio"]   = df["text"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    df["url_count"]     = df["text"].apply(lambda x: len(re.findall(r"http\S+|www\S+", x)))
    df["exclaim_count"] = df["text"].apply(lambda x: x.count("!"))
    return df


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PREPROCESAMIENTO DEL DATASET")
    print("=" * 60)

    # 1. Cargar datos
    df = pd.read_csv(RAW_CSV)
    print(f"\nRegistros cargados: {len(df)}")
    print(df["label"].value_counts())

    # 2. Codificar etiquetas: spam=1, ham=0
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])   # ham→0, spam→1
    print(f"\nClases: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 3. Limpieza de texto
    print("\nAplicando limpieza de texto...")
    df["text_clean"] = df["text"].apply(clean_text)

    # 4. Añadir características estructurales
    df = add_features(df)

    # 5. Split estratificado: 70% train, 15% val, 15% test
    X = df["text_clean"].values
    y = df["label_enc"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"\nTamaños de conjuntos:")
    print(f"  Entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validación:    {len(X_val)}  ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Prueba:        {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # 6. Vectorización TF-IDF (ajustada solo con train)
    print("\nEntrenando vectorizador TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),    # unigramas y bigramas
        sublinear_tf=True,     # aplicar log(tf) para reducir el efecto de términos frecuentes
        min_df=2               # ignorar términos que aparecen en < 2 documentos
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"Dimensiones TF-IDF: {X_train_tfidf.shape}")

    # 7. Guardar artefactos procesados
    print("\nGuardando artefactos...")
    joblib.dump(X_train_tfidf,  os.path.join(PROC_DIR, "X_train.joblib"))
    joblib.dump(X_val_tfidf,    os.path.join(PROC_DIR, "X_val.joblib"))
    joblib.dump(X_test_tfidf,   os.path.join(PROC_DIR, "X_test.joblib"))
    joblib.dump(y_train,        os.path.join(PROC_DIR, "y_train.joblib"))
    joblib.dump(y_val,          os.path.join(PROC_DIR, "y_val.joblib"))
    joblib.dump(y_test,         os.path.join(PROC_DIR, "y_test.joblib"))
    joblib.dump(vectorizer,     os.path.join(PROC_DIR, "vectorizer.joblib"))
    joblib.dump(le,             os.path.join(PROC_DIR, "label_encoder.joblib"))

    # Guardar también los textos crudos para modelos DL
    joblib.dump(X_train, os.path.join(PROC_DIR, "X_train_raw.joblib"))
    joblib.dump(X_val,   os.path.join(PROC_DIR, "X_val_raw.joblib"))
    joblib.dump(X_test,  os.path.join(PROC_DIR, "X_test_raw.joblib"))

    print("Preprocesamiento completado exitosamente.")


if __name__ == "__main__":
    main()

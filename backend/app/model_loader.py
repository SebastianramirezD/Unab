"""
backend/app/model_loader.py
---------------------------
Carga y cachea el modelo SVM y el vectorizador TF-IDF.
Utiliza un singleton para evitar cargas múltiples.
"""

import os
import re
import string
import joblib
import logging
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

STOP_WORDS = set(stopwords.words("english"))

# Rutas relativas al directorio de ejecución del backend
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.joblib")
VECT_PATH  = os.path.join(BASE_DIR, "datasets", "processed", "vectorizer.joblib")

_model      = None
_vectorizer = None


def load_model():
    global _model, _vectorizer
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Modelo no encontrado en {MODEL_PATH}. "
                           "Ejecuta scripts/training/train_ml_models.py primero.")
            return
        _model      = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECT_PATH)
        logger.info(f"Modelo SVM cargado desde {MODEL_PATH}")


def get_model():
    if _model is None:
        load_model()
    return _model, _vectorizer


def clean_text(text: str) -> str:
    """Misma limpieza que en preprocess.py para consistencia."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

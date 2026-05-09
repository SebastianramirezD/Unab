"""
backend/app/routers/predict.py
-------------------------------
Endpoints de predicción y explicación SHAP.

POST /api/predict      — Clasifica un mensaje
POST /api/explain      — Clasifica + retorna SHAP values
"""

import time
import shap
import logging
import numpy as np
from pydantic   import BaseModel, Field
from fastapi    import APIRouter, HTTPException
from app.model_loader import get_model, clean_text
from app.database     import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Esquemas Pydantic ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="Congratulations! You've won a FREE iPhone. Click now to claim!")

class PredictResponse(BaseModel):
    label:       str    # "spam" o "ham"
    probability: float  # probabilidad de ser spam
    confidence:  str    # "alta", "media", "baja"
    latency_ms:  float

class ExplainResponse(PredictResponse):
    top_features: list  # [{"feature": str, "shap_value": float}]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_confidence(prob: float) -> str:
    if prob >= 0.80 or prob <= 0.20:
        return "alta"
    if prob >= 0.65 or prob <= 0.35:
        return "media"
    return "baja"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse, summary="Clasificar un mensaje")
def predict(req: PredictRequest):
    """
    Clasifica un mensaje de texto como **spam** o **ham**.

    Retorna la etiqueta, la probabilidad de spam (0–1) y la confianza de la predicción.
    """
    model, vectorizer = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. "
                            "Ejecuta el script de entrenamiento primero.")

    t0        = time.perf_counter()
    clean     = clean_text(req.text)
    X         = vectorizer.transform([clean])
    prob      = float(model.predict_proba(X)[0, 1])
    label     = "spam" if prob >= 0.5 else "ham"
    elapsed   = (time.perf_counter() - t0) * 1000

    # Registrar predicción en BD para monitoreo
    log_prediction(text=req.text, label=label, probability=prob)

    return PredictResponse(
        label=label,
        probability=round(prob, 4),
        confidence=get_confidence(prob),
        latency_ms=round(elapsed, 2),
    )


@router.post("/explain", response_model=ExplainResponse,
             summary="Clasificar y explicar con SHAP")
def explain(req: PredictRequest):
    """
    Clasifica un mensaje y retorna las **10 características más influyentes**
    según SHAP (SHapley Additive exPlanations).

    - SHAP value > 0 → contribuye a clasificar como **spam**
    - SHAP value < 0 → contribuye a clasificar como **ham**
    """
    model, vectorizer = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    t0    = time.perf_counter()
    clean = clean_text(req.text)
    X     = vectorizer.transform([clean])
    prob  = float(model.predict_proba(X)[0, 1])
    label = "spam" if prob >= 0.5 else "ham"

    # SHAP con LinearExplainer (eficiente para SVM lineal calibrado)
    try:
        feature_names = vectorizer.get_feature_names_out()
        # Para KernelExplainer usamos una muestra del background
        # En producción se precalcula y cachea
        background = shap.sample(X, min(1, X.shape[0]))
        explainer  = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            background
        )
        sv = explainer.shap_values(X, nsamples=50)[0]  # shape (n_features,)

        # Top 10 características por |SHAP value|
        nz_idx   = X.nonzero()[1]  # solo características presentes en el mensaje
        top_idx  = nz_idx[np.argsort(np.abs(sv[nz_idx]))[-10:][::-1]]
        top_feats = [
            {"feature": feature_names[i], "shap_value": round(float(sv[i]), 5)}
            for i in top_idx
        ]
    except Exception as e:
        logger.warning(f"Error en SHAP: {e}")
        top_feats = []

    elapsed = (time.perf_counter() - t0) * 1000

    return ExplainResponse(
        label=label,
        probability=round(prob, 4),
        confidence=get_confidence(prob),
        latency_ms=round(elapsed, 2),
        top_features=top_feats,
    )

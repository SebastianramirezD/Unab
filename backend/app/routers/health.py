"""
backend/app/routers/health.py
"""
from fastapi import APIRouter
from app.model_loader import get_model

router = APIRouter()

@router.get("/health", summary="Estado del servicio")
def health():
    model, _ = get_model()
    return {
        "status":       "ok" if model is not None else "degradado",
        "model_loaded": model is not None,
        "version":      "1.0.0",
    }

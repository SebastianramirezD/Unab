"""
backend/app/routers/monitor.py
"""
from fastapi import APIRouter
from app.database import get_stats

router = APIRouter()

@router.get("/monitor/stats", summary="Estadísticas de predicciones")
def stats():
    """Retorna estadísticas agregadas de las predicciones realizadas."""
    return get_stats()

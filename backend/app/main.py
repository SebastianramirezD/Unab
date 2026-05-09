"""
backend/app/main.py
-------------------
API REST con FastAPI para el sistema de detección de spam.
Expone endpoints de predicción, explicación SHAP y monitoreo.
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict, health, monitor
from app.model_loader import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la aplicación."""
    logger.info("Cargando modelo SVM...")
    load_model()
    logger.info("Modelo cargado correctamente.")
    yield
    logger.info("Apagando servidor.")


app = FastAPI(
    title="Spam Detector API",
    description=(
        "API REST para clasificación de mensajes de spam/ham. "
        "Incluye predicción con probabilidades, análisis SHAP y monitoreo."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS (permite requests desde el frontend React en localhost:3000) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(predict.router,  prefix="/api", tags=["Predicción"])
app.include_router(health.router,   prefix="/api", tags=["Health"])
app.include_router(monitor.router,  prefix="/api", tags=["Monitoreo"])


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Spam Detector API v1.0",
        "docs":    "/docs",
        "health":  "/api/health",
    }

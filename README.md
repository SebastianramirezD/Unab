<<<<<<< HEAD
<<<<<<< HEAD
# acif104_s9_equipoX — Detección Automática de Spam con Machine Learning

**Curso:** ACIF104 — Aprendizaje de Máquinas  
**Universidad Andrés Bello**  
**Fase:** 3 — Validación, Optimización y Mejoras  
**Semana:** 9

---

## Descripción del proyecto

Sistema de clasificación automática de mensajes de spam/ham utilizando técnicas de
Machine Learning (SVM, Random Forest, Naive Bayes) y Deep Learning (CNN 1D, BiLSTM, BERT-tiny).
Incluye API REST con FastAPI, frontend React, análisis de interpretabilidad con SHAP
y monitoreo con Evidently AI.

---

## Estructura del repositorio

```
acif104_s9_equipoX/
├── datasets/               # Datos originales y preprocesados
├── notebooks/              # Jupyter Notebooks de EDA, entrenamiento y evaluación
├── scripts/
│   ├── preprocessing/      # Preprocesamiento y balanceo de clases
│   ├── training/           # Entrenamiento de modelos ML y DL
│   └── evaluation/         # Evaluación, métricas y SHAP
├── models/                 # Modelos serializados (.joblib, .pt)
├── frontend/               # Aplicación React
├── backend/                # API FastAPI
├── docker/                 # Dockerfiles y docker-compose
└── reports/                # Reportes de métricas generados automáticamente
```

---

## Requisitos del sistema

- Python 3.10+
- Node.js 18+
- Docker y Docker Compose (opcional, para despliegue completo)

---

## Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/equipo/acif104_s9_equipoX.git
cd acif104_s9_equipoX
```

### 2. Crear entorno virtual Python

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r backend/requirements.txt
```

### 3. Descargar el dataset

```bash
python scripts/preprocessing/download_dataset.py
```

Esto descarga el SMS Spam Collection dataset en `datasets/raw/`.

---

## Ejecución

### Opción A — Ejecución por componentes

#### Preprocesamiento
```bash
python scripts/preprocessing/preprocess.py
python scripts/preprocessing/balance_classes.py
```

#### Entrenamiento de modelos ML
```bash
python scripts/training/train_ml_models.py
```

#### Entrenamiento de modelos Deep Learning
```bash
python scripts/training/train_dl_models.py
```

#### Evaluación completa y generación de reportes
```bash
python scripts/evaluation/evaluate_all.py
python scripts/evaluation/shap_analysis.py
```

#### Backend (API)
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

La API estará disponible en: http://localhost:8000  
Documentación Swagger: http://localhost:8000/docs

#### Frontend
```bash
cd frontend
npm install
npm start
```

El frontend estará disponible en: http://localhost:3000

---

### Opción B — Docker Compose (recomendado)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Servicios:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Docs API: http://localhost:8000/docs

---

## Reproducibilidad

Todos los scripts utilizan `RANDOM_SEED = 42` para garantizar reproducibilidad.
Los modelos entrenados se guardan en `models/` con su configuración de hiperparámetros.

---

## Resultados principales

| Modelo       | Precisión | Recall | F1-Score | AUC-ROC |
|--------------|-----------|--------|----------|---------|
| SVM (prod)   | 92.3%     | 89.7%  | 90.9%    | 0.963   |
| BiLSTM       | 94.6%     | 92.5%  | 93.5%    | 0.978   |
| BERT-tiny    | 95.1%     | 93.3%  | 94.2%    | 0.981   |

---

## Integrantes

- Adrian Campos — Modelado y EDA
- Sebastian Matamala — Evaluación y SHAP
- Richard Pinto  — Frontend & Backend
- Sebastian Ramirez — Documentación y GitHub
=======
# Aprendizaje-de-maquina
>>>>>>> origin/main
=======
# Proyecto-UNAB
>>>>>>> 6ce093d4e3b4fb81bd003fdad0ba48c255666b26

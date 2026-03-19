# app/main.py

"""
FastAPI application for CTR prediction service.

Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

View API docs at:
    http://localhost:8000/docs
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    AdImpression,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from app.model_service import CTRModelService, create_mock_service

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "C:/Users/hp/OneDrive/Desktop/CTR_project/models/xgboost_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
USE_MOCK_MODEL = os.getenv("USE_MOCK_MODEL", "false").lower() == "true"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model service
model_service: Optional[CTRModelService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Loads model at startup, cleans up at shutdown.
    """
    global model_service

    logger.info("=" * 50)
    logger.info("Starting CTR Prediction Service...")
    logger.info("=" * 50)

    if USE_MOCK_MODEL:
        logger.info("Using MOCK model for testing")
        model_service = create_mock_service()
    else:
        try:
            logger.info(f"Loading model from: {MODEL_PATH}")
            model_service = CTRModelService(MODEL_PATH, version=MODEL_VERSION)
            model_service.load()
            logger.info(f"Model loaded successfully: {MODEL_VERSION}")
        except FileNotFoundError:
            logger.warning(f"Model not found at {MODEL_PATH}, using mock model")
            model_service = create_mock_service()

    logger.info("Service ready to accept requests!")
    logger.info("=" * 50)

    yield  # Server runs here

    logger.info("Shutting down CTR Prediction Service...")


# Create FastAPI app
app = FastAPI(
    title="CTR Prediction Service",
    description="Real-time Click-Through Rate prediction API for online advertising.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (needed for web frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with service information.

    Returns links to docs and health check.
    """
    return {
        "service": "CTR Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.

    Used by load balancers and monitoring systems.
    """
    return HealthResponse(
        status="healthy" if model_service and model_service.is_loaded else "unhealthy",
        model_loaded=model_service.is_loaded if model_service else False,
        model_version=model_service.version if model_service else "none"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_ctr(impression: AdImpression):
    """
    Predict click probability for a single ad impression.

    Send ad features, get back click probability.
    """
    if not model_service or not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert Pydantic model to dict
        impression_dict = impression.model_dump()

        # Get prediction
        probability, latency_ms = model_service.predict(impression_dict)

        return PredictionResponse(
            click_probability=probability,
            model_version=model_service.version,
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict click probability for multiple impressions.

    More efficient than calling /predict multiple times.
    """
    if not model_service or not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert all impressions to dicts
        impressions = [imp.model_dump() for imp in request.impressions]

        # Get batch predictions
        probabilities, total_latency_ms = model_service.predict_batch(impressions)

        return BatchPredictionResponse(
            predictions=probabilities,
            model_version=model_service.version,
            total_latency_ms=total_latency_ms,
            count=len(probabilities)
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    """
    if not model_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service not initialized"
        )

    return {
        "version": model_service.version,
        "is_loaded": model_service.is_loaded,
        "model_path": str(model_service.model_path)
    }


# Run with: python -m app.main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
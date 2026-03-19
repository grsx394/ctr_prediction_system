# app/schemas.py

"""
Pydantic schemas for CTR prediction API.

These define the contract for your API:
- What inputs the API accepts
- What outputs the API returns
- Automatic validation of requests
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List


class AdImpression(BaseModel):
    """
    Single ad impression for CTR prediction.

    These fields match the features our model was trained on.
    """

    # Original features from dataset
    C14: int = Field(..., description="Anonymous feature (ad identifier)")
    C1: int = Field(default=0, description="Anonymous categorical variable")
    C15: int = Field(default=0, description="Anonymous categorical variable")
    C16: int = Field(default=0, description="Anonymous categorical variable")
    C17: int = Field(default=0, description="Anonymous categorical variable")
    C18: int = Field(default=0, description="Anonymous categorical variable")
    C19: int = Field(default=0, description="Anonymous categorical variable")
    C21: int = Field(default=0, description="Anonymous categorical variable")

    # Device features
    device_type: int = Field(default=0, ge=0, le=4, description="Device type (0-4)")
    device_conn_type: int = Field(default=0, ge=0, le=4, description="Connection type")

    # Ad position
    banner_pos: int = Field(default=0, ge=0, description="Banner position on page")

    # Temporal features (user provides these)
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")

    # Categorical features (as strings, we'll encode them)
    site_category: str = Field(default="unknown", description="Site category")
    app_category: str = Field(default="unknown", description="App category")
    site_domain: str = Field(default="unknown", description="Site domain")
    app_domain: str = Field(default="unknown", description="App domain")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "C14": 15706,
                "C1": 1005,
                "C15": 320,
                "C16": 50,
                "C17": 1722,
                "C18": 0,
                "C19": 35,
                "C21": 79,
                "device_type": 1,
                "device_conn_type": 0,
                "banner_pos": 0,
                "hour": 14,
                "day_of_week": 2,
                "site_category": "28905ebd",
                "app_category": "07d7df22",
                "site_domain": "f3845767",
                "app_domain": "7801e8d9"
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response from single CTR prediction."""

    click_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of click (0-1)"
    )
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Inference time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Request for multiple predictions at once."""

    impressions: List[AdImpression] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of impressions (max 1000)"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[float] = Field(..., description="Click probabilities")
    model_version: str = Field(..., description="Model version used")
    total_latency_ms: float = Field(..., description="Total processing time")
    count: int = Field(..., description="Number of predictions made")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Is model loaded")
    model_version: str = Field(..., description="Model version")
# app/model_service.py

"""
Model service for CTR prediction.

Handles:
1. Loading the trained model from disk
2. Feature transformation (must match training!)
3. Making predictions
"""

import pickle
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CTRModelService:
    """
    Service class for CTR model inference.

    Loads model once at startup, reuses for all requests.
    """

    def __init__(self, model_path: str, version: str = "v1.0.0"):
        """
        Initialize the model service.

        Args:
            model_path: Path to the saved model (.pkl file)
            version: Version string for tracking
        """
        self.model_path = Path(model_path)
        self.version = version
        self.model = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Load the model from disk.

        Call this once at application startup.
        """
        logger.info(f"Loading model from {self.model_path}")
        start_time = time.time()

        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

            self._is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if model is ready for inference."""
        return self._is_loaded and self.model is not None

    def _transform_features(self, impression: dict) -> pd.DataFrame:
        """
        Transform raw impression data into model features.

        IMPORTANT: This must match the feature engineering in features.py!

        Args:
            impression: Dictionary of raw impression features

        Returns:
            DataFrame with features in correct order
        """
        # Create feature dictionary matching training order
        features = {
            'C14': impression.get('C14', 0),
            'hour_of_day': impression.get('hour', 0),
            'day_of_week': impression.get('day_of_week', 0),
            'C1': impression.get('C1', 0),
            'banner_pos': impression.get('banner_pos', 0),
            'device_type': impression.get('device_type', 0),
            'device_conn_type': impression.get('device_conn_type', 0),
            'C15': impression.get('C15', 0),
            'C16': impression.get('C16', 0),
            'C17': impression.get('C17', 0),
            'C18': impression.get('C18', 0),
            'C19': impression.get('C19', 0),
            'C21': impression.get('C21', 0),
            # Encode categorical features using hash
            'site_category_encoded': hash(impression.get('site_category', 'unknown')) % 10000,
            'app_category_encoded': hash(impression.get('app_category', 'unknown')) % 10000,
            'site_domain_encoded': hash(impression.get('site_domain', 'unknown')) % 10000,
            'app_domain_encoded': hash(impression.get('app_domain', 'unknown')) % 10000,
        }

        # Create DataFrame with single row
        df = pd.DataFrame([features])

        return df

    def _transform_batch(self, impressions: list) -> pd.DataFrame:
        """Transform multiple impressions into feature matrix."""
        dfs = [self._transform_features(imp) for imp in impressions]
        return pd.concat(dfs, ignore_index=True)

    def predict(self, impression: dict) -> tuple:
        """
        Predict CTR for a single impression.

        Args:
            impression: Dictionary of impression features

        Returns:
            Tuple of (probability, latency_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        # Transform features
        features = self._transform_features(impression)

        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0, 1]
        else:
            proba = float(self.model.predict(features)[0])

        latency_ms = (time.time() - start_time) * 1000

        return float(proba), latency_ms

    def predict_batch(self, impressions: list) -> tuple:
        """
        Predict CTR for multiple impressions.

        Args:
            impressions: List of impression dictionaries

        Returns:
            Tuple of (list of probabilities, total_latency_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        # Transform all features
        features = self._transform_batch(impressions)

        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(features)[:, 1]
        else:
            probas = self.model.predict(features)

        latency_ms = (time.time() - start_time) * 1000

        return probas.tolist(), latency_ms


class MockCTRModel:
    """Mock model for testing without a real model."""

    def predict_proba(self, X) -> np.ndarray:
        """Return random probabilities for testing."""
        n_samples = len(X)
        probas = np.random.uniform(0.1, 0.3, size=n_samples)
        return np.column_stack([1 - probas, probas])


def create_mock_service() -> CTRModelService:
    """Create a model service with mock model for testing."""

    # Create mock model file
    mock_path = Path("models/mock_model.pkl")
    mock_path.parent.mkdir(exist_ok=True)

    with open(mock_path, 'wb') as f:
        pickle.dump(MockCTRModel(), f)

    service = CTRModelService(str(mock_path), version="mock-v1.0.0")
    service.load()

    return service
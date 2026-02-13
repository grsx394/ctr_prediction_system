# src/model.py

import pickle
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


class CTRModel:
    """
    Logistic Regression model for CTR prediction.

    Why Logistic Regression:
    - Outputs probabilities (0 to 1) — perfect for CTR
    - Fast to train
    - Works well with numeric features
    - Easy to understand and debug
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,  # Enough iterations to converge
            solver='lbfgs',  # Good default solver
            random_state=42  # Reproducible results
        )

    def fit(self, X, y):
        """
        Train the model on features and labels.

        X: feature matrix (from prepare_features)
        y: labels (0 or 1)
        """
        self.model.fit(X, y)
        print(f"CTRModel trained on {len(X)} samples with {X.shape[1]} features")
        return self

    def predict(self, X):
        """
        Predict click probabilities.

        Returns: array of probabilities (0.0 to 1.0)
        """
        # predict_proba returns [[prob_class_0, prob_class_1], ...]
        # We want prob_class_1 (probability of click)
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def save(self, filepath):
        """
        Save model to disk.

        Why: So we can load it later for predictions without retraining.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load model from disk.

        Why: For inference on new data.
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return self


class XGBoostCTRModel:
    """
    XGBoost model for CTR prediction.

    Why XGBoost:
    - Handles complex feature interactions
    - Often achieves better performance than linear models
    - Built-in handling of missing values
    - Fast training with parallel processing
    """

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,  # Number of trees
            max_depth=5,  # Depth of each tree (prevents overfitting)
            learning_rate=0.1,  # Step size for updates
            objective='binary:logistic',  # Binary classification with probabilities
            random_state=42,  # Reproducible results
            eval_metric='logloss',  # Metric to optimize
         )

    def fit(self, X, y):
        """
        Train the XGBoost model.

        X: feature matrix
        y: labels (0 or 1)
        """
        self.model.fit(X, y)
        print(f"XGBoostCTRModel trained on {len(X)} samples with {X.shape[1]} features")
        return self

    def predict(self, X):
        """
        Predict click probabilities.

        Returns: array of probabilities (0.0 to 1.0)
        """
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def save(self, filepath):
        """
        Save model to disk.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load model from disk.
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return self
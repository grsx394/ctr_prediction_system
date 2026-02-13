# train.py

"""
Main training script for CTR prediction system.

Usage:
    python train.py

What it does:
    1. Loads raw training data
    2. Splits into train/validation
    3. Prepares features
    4. Trains baselines and ML models
    5. Evaluates all models
    6. Saves the best model
"""

import sys

sys.path.append('src')

from dataloader import load_data, validate_data, train_val_split
from features import prepare_features
from baselines import GlobalCTRBaseline, PerAdCTRBaseline
from model import CTRModel, XGBoostCTRModel
from evaluation import evaluate_model, compare_models

# === CONFIGURATION ===
DATA_PATH = 'C:/Users/hp/OneDrive/Desktop/CTR_project/data/raw/train.csv'
MODEL_PATH = 'models/ctr_model.pkl'
XGBOOST_MODEL_PATH = 'models/xgboost_model.pkl'
N_ROWS = 100000  # Use 100k rows for training (increase for better results)


def main():
    print("=" * 60)
    print("CTR PREDICTION SYSTEM - TRAINING")
    print("=" * 60)

    # --- Step 1: Load Data ---
    print("\n[1/6] Loading data...")
    df = load_data(DATA_PATH, nrows=N_ROWS)
    validate_data(df)

    # --- Step 2: Split Data ---
    print("\n[2/6] Splitting data...")
    train_df, val_df = train_val_split(df, val_ratio=0.2)

    # --- Step 3: Prepare Features ---
    print("\n[3/6] Preparing features...")
    X_train, y_train = prepare_features(train_df, fit_scaler=True)
    X_val, y_val = prepare_features(val_df, fit_scaler=False)

    # --- Step 4: Train Models ---
    print("\n[4/6] Training models...")

    # Baseline 1: Global CTR
    global_baseline = GlobalCTRBaseline()
    global_baseline.fit(X_train, y_train)

    # Baseline 2: Per-Ad CTR
    ad_baseline = PerAdCTRBaseline()
    ad_baseline.fit(X_train, y_train)

    # ML Model 1: Logistic Regression
    lr_model = CTRModel()
    lr_model.fit(X_train, y_train)

    # ML Model 2: XGBoost
    xgb_model = XGBoostCTRModel()
    xgb_model.fit(X_train, y_train)

    # --- Step 5: Evaluate Models ---
    print("\n[5/6] Evaluating models...")

    global_preds = global_baseline.predict(X_val)
    ad_preds = ad_baseline.predict(X_val)
    lr_preds = lr_model.predict(X_val)
    xgb_preds = xgb_model.predict(X_val)

    results = []
    results.append(evaluate_model(y_val, global_preds, "Global CTR"))
    results.append(evaluate_model(y_val, ad_preds, "Per-Ad CTR"))
    results.append(evaluate_model(y_val, lr_preds, "Logistic Regression"))
    results.append(evaluate_model(y_val, xgb_preds, "XGBoost"))

    compare_models(results)

    # --- Step 6: Save Models ---
    print("\n[6/6] Saving models...")

    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)

    lr_model.save(MODEL_PATH)
    xgb_model.save(XGBOOST_MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
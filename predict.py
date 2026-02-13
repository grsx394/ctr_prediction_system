# predict.py

"""
Prediction script for CTR prediction system.

Usage:
    python predict.py

What it does:
    1. Loads saved model
    2. Loads new data for prediction
    3. Prepares features
    4. Generates CTR predictions
    5. Saves predictions to CSV
"""

import sys

sys.path.append('src')

import pandas as pd
from dataloader import load_data
from features import prepare_features
from model import CTRModel

# === CONFIGURATION ===
MODEL_PATH = 'models/ctr_model.pkl'
INPUT_PATH = 'C:/Users/hp/OneDrive/Desktop/CTR_project/data/train.gz'  # Using train.csv for demo
OUTPUT_PATH = 'predictions/predictions.csv'
N_ROWS = 1000  # Predict on 1000 rows for demo


def main():
    print("=" * 60)
    print("CTR PREDICTION SYSTEM - INFERENCE")
    print("=" * 60)

    # --- Step 1: Load Model ---
    print("\n[1/4] Loading saved model...")
    model = CTRModel()
    model.load(MODEL_PATH)

    # --- Step 2: Load New Data ---
    print("\n[2/4] Loading data for prediction...")
    df = load_data(INPUT_PATH, nrows=N_ROWS)
    print(f"Loaded {len(df)} rows")

    # --- Step 3: Prepare Features ---
    print("\n[3/4] Preparing features...")
    # Note: fit_scaler=True because we need to fit on this data
    # In production, you would save/load the scaler separately
    X, y = prepare_features(df, fit_scaler=True)

    # --- Step 4: Generate Predictions ---
    print("\n[4/4] Generating predictions...")
    predictions = model.predict(X)

    # Create output DataFrame
    output_df = pd.DataFrame({
        'id': df['id'],
        'predicted_ctr': predictions
    })

    # Save predictions
    import os
    os.makedirs('predictions', exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nPredictions saved to {OUTPUT_PATH}")
    print(f"Sample predictions:")
    print(output_df.head(10))

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
# test_model.py

import sys
sys.path.append('src')

from dataloader import load_data, train_val_split
from features import prepare_features
from baselines import GlobalCTRBaseline, PerAdCTRBaseline
from model import CTRModel
from evaluation import evaluate_model, compare_models

# Load and prepare data
df = load_data('C:/Users/hp/OneDrive/Desktop/CTR_project/data/train.gz', nrows=10000)
train_df, val_df = train_val_split(df)

X_train, y_train = prepare_features(train_df, fit_scaler=True)
X_val, y_val = prepare_features(val_df, fit_scaler=False)

# Train baselines
global_baseline = GlobalCTRBaseline()
global_baseline.fit(X_train, y_train)

ad_baseline = PerAdCTRBaseline()
ad_baseline.fit(X_train, y_train)

# Train ML model
print("\n--- Training ML Model ---")
ml_model = CTRModel()
ml_model.fit(X_train, y_train)

# Get predictions
global_preds = global_baseline.predict(X_val)
ad_preds = ad_baseline.predict(X_val)
ml_preds = ml_model.predict(X_val)

# Show sample predictions
print(f"\nSample ML predictions: {ml_preds[:5]}")

# Evaluate all models
print("\n--- Evaluation Results ---\n")
results = []

results.append(evaluate_model(y_val, global_preds, "Global CTR"))
print()
results.append(evaluate_model(y_val, ad_preds, "Per-Ad CTR"))
print()
results.append(evaluate_model(y_val, ml_preds, "Logistic Regression"))

# Compare all models
compare_models(results)
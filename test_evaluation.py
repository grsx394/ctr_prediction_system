# test_evaluation.py

import sys
sys.path.append('src')

from dataloader import load_data, train_val_split
from features import prepare_features
from baselines import GlobalCTRBaseline, PerAdCTRBaseline
from evaluation import evaluate_model, compare_models

# Load and prepare data
df = load_data('C:/Users/hp/OneDrive/Desktop/CTR_project/data/train.gz', nrows=10000)
train_df, val_df = train_val_split(df)

X_train, y_train = prepare_features(train_df)
X_val, y_val = prepare_features(val_df)

# Train baselines
global_baseline = GlobalCTRBaseline()
global_baseline.fit(X_train, y_train)

ad_baseline = PerAdCTRBaseline()
ad_baseline.fit(X_train, y_train)

# Get predictions on validation set
global_preds = global_baseline.predict(X_val)
ad_preds = ad_baseline.predict(X_val)

# Evaluate each model
print("\n--- Evaluation Results ---\n")
results = []

result1 = evaluate_model(y_val, global_preds, "Global CTR")
results.append(result1)

print()  # Empty line between models

result2 = evaluate_model(y_val, ad_preds, "Per-Ad CTR")
results.append(result2)

# Compare all models
compare_models(results)
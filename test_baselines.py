# test_baselines.py

import sys
sys.path.append('src')

from dataloader import load_data, train_val_split
from features import prepare_features
from baselines import GlobalCTRBaseline, PerAdCTRBaseline

# Load and prepare data
df = load_data('C:/Users/hp/OneDrive/Desktop/CTR_project/data/train.gz', nrows=10000)
train_df, val_df = train_val_split(df)

X_train, y_train = prepare_features(train_df)
X_val, y_val = prepare_features(val_df)

# Test Global CTR Baseline
print("--- Global CTR Baseline ---")
global_baseline = GlobalCTRBaseline()
global_baseline.fit(X_train, y_train)
global_preds = global_baseline.predict(X_val)
print(f"Sample predictions: {global_preds[:5]}")

# Test Per-Ad CTR Baseline
print("\n--- Per-Ad CTR Baseline ---")
ad_baseline = PerAdCTRBaseline()
ad_baseline.fit(X_train, y_train)
ad_preds = ad_baseline.predict(X_val)
print(f"Sample predictions: {ad_preds[:5]}")
print(f"Unique prediction values: {len(set(ad_preds))}")
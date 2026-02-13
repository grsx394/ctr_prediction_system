# test_features.py

import sys
sys.path.append('src')

from dataloader import load_data, train_val_split
from features import prepare_features

# Load data
df = load_data('C:/Users/hp/OneDrive/Desktop/CTR_project/data/train.gz', nrows=10000)

# Split
train_df, val_df = train_val_split(df)

# Prepare features for training set
print("--- Training Features ---")
X_train, y_train = prepare_features(train_df)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"\nFeature columns: {X_train.columns.tolist()}")
print(f"\nFirst 3 rows:\n{X_train.head(3)}")

# Prepare features for validation set
print("\n--- Validation Features ---")
X_val, y_val = prepare_features(val_df)
print(f"X_val shape: {X_val.shape}")
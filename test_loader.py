# test_loader.py

# This line lets Python find modules inside the 'src' folder
import sys
sys.path.append('src')

# import data_loader module
from dataloader import load_data, validate_data, train_val_split

# Step 1: Load a small sample of data
print("--- Loading Data ---")
df = load_data('C:/Users/hp/OneDrive/Desktop/CTR_project/data/  train.gz', nrows=10000)
print(f"Loaded {len(df)} rows")

# Step 2: Validate the data
print("\n--- Validating Data ---")
validate_data(df)

# Step 3: Split into train and validation
print("\n--- Splitting Data ---")
train_df, val_df = train_val_split(df, val_ratio=0.2)

# Step 4: Verify the split
print("\n--- Verification ---")
print(f"Train clicks: {train_df['click'].sum()} / {len(train_df)}")
print(f"Val clicks: {val_df['click'].sum()} / {len(val_df)}")
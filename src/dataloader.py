import pandas as pd

def load_data(filepath, nrows=None):
    df = pd.read_csv('C:/Users/hp/OneDrive/Desktop/CTR_project/data/train.gz', nrows= nrows)  #Load raw CTR data from CSV file.
    return df

def validate_data(df):
    required_columns = ['click', 'hour', 'banner_pos', 'device_type', 'device_conn_type']  #Check that required columns exist and label has no missing values.

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")  # If any columns are missing, we stop the program

    if df['click'].isnull().any():
        raise ValueError("Label column 'click' has missing values")  #check if the click column has any empty values

    print(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True

def train_val_split(df, val_ratio=0.2, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle rows
    split_idx = int(len(df) * (1 - val_ratio))  # Calculate split point

    train_df = df[:split_idx]  #rows from start up to (not including) split_idx
    val_df = df[split_idx:]  #rows from split_idx to the end

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
    return train_df, val_df

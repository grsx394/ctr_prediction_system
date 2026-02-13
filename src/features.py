# src/features.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Global scaler — will be fitted on training data
scaler = None

# Global encoders for categorical features
encoders = {}


def extract_time_features(df):
    """
    Extract hour of day and day of week from the 'hour' column.

    Why: CTR varies by time — people click more at certain hours/days.

    Input format: 'hour' column contains values like 14102100
        - 14 = year (2014)
        - 10 = month (October)
        - 21 = day
        - 00 = hour (midnight)

    Output: Two new columns added to df
        - 'hour_of_day' (0-23)
        - 'day_of_week' (0-6, where 0=Monday)
    """
    # Convert to string to extract parts
    hour_str = df['hour'].astype(str)

    # Extract hour of day (last 2 digits)
    df['hour_of_day'] = hour_str.str[-2:].astype(int)

    # Extract full date to get day of week
    # Format: YYMMDDHH -> we need YYMMDD
    date_str = hour_str.str[:-2]  # Remove last 2 digits (hour)
    df['date'] = pd.to_datetime(date_str, format='%y%m%d')
    df['day_of_week'] = df['date'].dt.dayofweek

    # Drop the temporary date column
    df = df.drop(columns=['date'])

    return df


def encode_categorical_features(df, fit=False):
    """
    Convert categorical string columns to numeric using Label Encoding.

    Why: ML models need numeric input. Label encoding assigns each unique
    category a number (e.g., 'sports' -> 0, 'news' -> 1, etc.)

    fit=True: Learn encoding from this data (use for training)
    fit=False: Apply existing encoding (use for validation/test)
    """
    global encoders

    categorical_columns = ['site_category', 'app_category', 'site_domain', 'app_domain']

    for col in categorical_columns:
        if fit:
            # Create and fit new encoder
            encoders[col] = LabelEncoder()
            # Handle unseen categories by adding 'unknown'
            df[col] = df[col].astype(str)
            encoders[col].fit(df[col])
            df[col + '_encoded'] = encoders[col].transform(df[col])
        else:
            # Use existing encoder, handle unseen values
            df[col] = df[col].astype(str)
            # Replace unseen categories with the most frequent one from training
            known_categories = set(encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_categories else encoders[col].classes_[0])
            df[col + '_encoded'] = encoders[col].transform(df[col])

    return df


def select_features(df):
    """
    Select which columns to use as features for the model.

    Why: Not all columns are useful. We pick:
        - Numeric features that are already usable
        - Time features we extracted
        - Encoded categorical features

    Returns: X (features), y (label)
    """
    feature_columns = [
        # Identifiers
        'C14',  # Ad identifier (needed for Per-Ad baseline)

        # Time features (extracted)
        'hour_of_day',
        'day_of_week',

        # Original numeric features
        'C1',
        'banner_pos',
        'device_type',
        'device_conn_type',
        'C15',
        'C16',
        'C17',
        'C18',
        'C19',
        'C21',

        # New: Encoded categorical features
        'site_category_encoded',
        'app_category_encoded',
        'site_domain_encoded',
        'app_domain_encoded',
    ]

    X = df[feature_columns]
    y = df['click']

    return X, y


def scale_features(X, fit=False):
    """
    Scale features to have zero mean and unit variance.

    Why:
    - ML models work better when features are on similar scales
    - Prevents features with large values from dominating
    - Fixes convergence warnings in Logistic Regression

    fit=True: Learn scaling from this data (use for training)
    fit=False: Apply existing scaling (use for validation/test)
    """
    global scaler

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Convert back to DataFrame to keep column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X_scaled


def prepare_features(df, fit_scaler=False):
    """
    Full feature preparation pipeline.

    Why: Combines all feature steps into one function.
    This makes it easy to apply the same processing to train and validation data.

    Steps:
        1. Extract time features
        2. Encode categorical features
        3. Select model features
        4. Scale features

    fit_scaler=True: Use for training data
    fit_scaler=False: Use for validation/test data

    Returns: X (features), y (label)
    """
    df = df.copy()  # Avoid modifying original
    df = extract_time_features(df)
    df = encode_categorical_features(df, fit=fit_scaler)
    X, y = select_features(df)
    X = scale_features(X, fit=fit_scaler)

    print(f"Features prepared: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y
# CTR Prediction System

An end-to-end Click-Through Rate (CTR) prediction system for online advertising.

## Project Overview

**Course:** CS 5130  
**Goal:** Build a well-engineered ML system, not just a model

This project predicts the probability that a user will click on an advertisement based on ad impression features. The focus is on engineering quality, reproducibility, and proper evaluation — not state-of-the-art performance.

## System Architecture
```
Raw Ad Logs (CSV)
       ↓
   Data Loader (validation, splitting)
       ↓
   Feature Engineering (time features, scaling)
       ↓
   ┌─────────────────────────────────────┐
   │  Baselines          │  ML Model     │
   │  - Global CTR       │  - Logistic   │
   │  - Per-Ad CTR       │    Regression │
   └─────────────────────────────────────┘
       ↓
   Evaluation (AUC-ROC, Log Loss)
       ↓
   Predictions (CSV)
```

## Results

| Model              | AUC-ROC | Log Loss | Notes |
|--------------------|---------|----------|-------|
| Global CTR         | 0.5000  | 0.4603   | Baseline (random) |
| Per-Ad CTR         | 0.6924  | 0.4281   | Strong non-ML baseline |
| Logistic Regression| 0.6524  | 0.4353   | Simple ML model |
| **XGBoost**        | **0.7172** | **0.4129** | **Best model** |

**Key Findings:**
1. XGBoost outperforms all baselines and Logistic Regression on both metrics
2. Per-Ad CTR is a surprisingly strong baseline — simple historical aggregates work well
3. Adding categorical features (site_category, app_category) improved model performance

## Project Structure
```
ctr-prediction/
├── data/
│   └── raw/              # Raw dataset (train.csv)
├── models/               # Saved trained models
├── predictions/          # Output predictions
├── src/
│   ├── data_loader.py    # Data loading and validation
│   ├── features.py       # Feature engineering
│   ├── baselines.py      # Baseline models
│   ├── model.py          # ML model
│   └── evaluation.py     # Metrics and comparison
├── train.py              # Training script
├── predict.py            # Prediction script
└── README.md
```

## How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### 2. Download Dataset

Download the Avazu CTR dataset from Kaggle:
https://www.kaggle.com/c/avazu-ctr-prediction/data

Place `train.csv` in `data/raw/`

### 3. Train Models
```bash
python train.py
```

This will:
- Load and validate data
- Train baselines and ML model
- Evaluate all models
- Save trained model to `models/`

### 4. Generate Predictions
```bash
python predict.py
```

This will:
- Load saved model
- Generate predictions on new data
- Save predictions to `predictions/`

## Features Used

| Feature              | Type        | Description                    |
|----------------------|-------------|--------------------------------|
| hour_of_day          | Extracted   | Hour (0-23) from timestamp     |
| day_of_week          | Extracted   | Day (0-6) from timestamp       |
| C14                  | Numeric     | Anonymous feature (ad-related) |
| banner_pos           | Numeric     | Ad position on page            |
| device_type          | Numeric     | Device category                |
| device_conn_type     | Numeric     | Connection type                |
| C1, C15-C19, C21     | Numeric     | Anonymous features             |
| site_category_encoded| Categorical | Website category (encoded)     |
| app_category_encoded | Categorical | App category (encoded)         |
| site_domain_encoded  | Categorical | Website domain (encoded)       |
| app_domain_encoded   | Categorical | App domain (encoded)           |

**Total: 17 features**

## Evaluation Metrics

- **AUC-ROC:** Measures ranking quality (0.5 = random, 1.0 = perfect)
- **Log Loss:** Measures probability calibration (lower = better)

## Trade-offs Identified

1. **AUC vs Log Loss:** Per-Ad CTR has better AUC, but Logistic Regression may have better calibrated probabilities for specific use cases.

2. **Simplicity vs Generalization:** Per-Ad CTR memorizes historical performance, which works well for known ads but fails for new ads (cold start problem).

3. **Feature Engineering vs Model Complexity:** Simple features with a simple model can match or beat complex approaches.

## Ethical Considerations

**Risk:** CTR optimization can lead to bias amplification — popular ads get more exposure, creating a feedback loop.

**Mitigation:** Implement exploration strategies (e.g., epsilon-greedy) to ensure new ads get fair exposure regardless of predicted CTR.

## Author

Gaurang Sathe
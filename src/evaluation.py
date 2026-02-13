# src/evaluation.py

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def calculate_auc(y_true, y_pred):
    """
    Calculate AUC-ROC (Area Under the ROC Curve).

    Why AUC:
    - Measures ranking quality: do we rank clicks higher than non-clicks?
    - Works well for imbalanced data (unlike accuracy)
    - 0.5 = random guessing, 1.0 = perfect

    y_true: actual labels (0 or 1)
    y_pred: predicted probabilities (0.0 to 1.0)
    """
    auc = roc_auc_score(y_true, y_pred)
    return auc


def calculate_log_loss(y_true, y_pred):
    """
    Calculate Log Loss (Cross-Entropy Loss).

    Why Log Loss:
    - Measures how confident AND correct predictions are
    - Penalizes confident wrong predictions heavily
    - Lower is better (0 = perfect)

    y_true: actual labels (0 or 1)
    y_pred: predicted probabilities (0.0 to 1.0)
    """
    # Clip predictions to avoid log(0) which gives infinity
    y_pred_clipped = np.clip(y_pred, 0.0001, 0.9999)
    ll = log_loss(y_true, y_pred_clipped)
    return ll


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculate and print all metrics for a model.

    Why this function:
    - Consistent evaluation across all models
    - Easy comparison
    - Returns metrics as dictionary for later use

    y_true: actual labels
    y_pred: predicted probabilities
    model_name: name to display in output
    """
    auc = calculate_auc(y_true, y_pred)
    ll = calculate_log_loss(y_true, y_pred)

    print(f"{model_name}:")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")

    return {'model': model_name, 'auc': auc, 'log_loss': ll}


def compare_models(results_list):
    """
    Print a comparison table of all models.

    Why this function:
    - Shows all results side by side
    - Makes it easy to see which model is best

    results_list: list of dictionaries from evaluate_model()
    """
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"{'Model':<20} {'AUC-ROC':<12} {'Log Loss':<12}")
    print("-" * 50)

    for result in results_list:
        print(f"{result['model']:<20} {result['auc']:<12.4f} {result['log_loss']:<12.4f}")

    print("=" * 50)
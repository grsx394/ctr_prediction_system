# src/baselines.py

import pandas as pd
import numpy as np


class GlobalCTRBaseline:
    """
    Baseline 1: Predict the same global CTR for all impressions.

    Why: This is the simplest possible model.
    It just calculates: total clicks / total impressions
    and predicts that value for everyone.

    If our ML model can't beat this, something is very wrong.
    """

    def __init__(self):
        self.global_ctr = None

    def fit(self, X, y):
        """
        Learn the global CTR from training data.

        X: features (not used, but included for consistent interface)
        y: labels (0 or 1)
        """
        self.global_ctr = y.mean()
        print(f"GlobalCTR trained: CTR = {self.global_ctr:.4f}")
        return self

    def predict(self, X):
        """
        Predict the same global CTR for all rows.

        Returns: array of predictions (all same value)
        """
        return np.full(len(X), self.global_ctr)


class PerAdCTRBaseline:
    """
    Baseline 2: Predict based on each ad's historical CTR.

    Why: Some ads perform better than others.
    This baseline remembers CTR for each ad (using C14 as ad identifier).
    For unseen ads, it falls back to global CTR.

    This is a strong baseline — often hard to beat significantly.
    """

    def __init__(self):
        self.ad_ctr = {}
        self.global_ctr = None

    def fit(self, X, y):
        """
        Learn CTR for each ad from training data.

        Steps:
        1. Calculate global CTR (fallback)
        2. Group by ad ID and calculate CTR per ad
        """
        # Global CTR as fallback
        self.global_ctr = y.mean()

        # Combine features and label for grouping
        df = X.copy()
        df['click'] = y.values

        # Calculate CTR per ad (using C14 as ad identifier)
        ad_stats = df.groupby('C14')['click'].agg(['sum', 'count'])
        ad_stats['ctr'] = ad_stats['sum'] / ad_stats['count']

        # Store as dictionary: ad_id -> ctr
        self.ad_ctr = ad_stats['ctr'].to_dict()

        print(f"PerAdCTR trained: {len(self.ad_ctr)} unique ads, fallback CTR = {self.global_ctr:.4f}")
        return self

    def predict(self, X):
        """
        Predict CTR based on ad ID.

        For each row:
        - If ad was seen in training: use its CTR
        - If ad is new: use global CTR
        """
        predictions = []
        for ad_id in X['C14']:
            if ad_id in self.ad_ctr:
                predictions.append(self.ad_ctr[ad_id])
            else:
                predictions.append(self.global_ctr)

        return np.array(predictions)
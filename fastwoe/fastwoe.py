"""fastwoe.py."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder


class FastWoe:  # pylint: disable=invalid-name
    """
    Fast Weight of Evidence (WOE) Encoder using scikit-learn's TargetEncoder.
    Stores mapping tables for each categorical feature, including:
    - Category value
    - Number of observations (count)
    - Average target (event rate)
    - WOE value
    - Feature-level statistics (Gini, IV, etc.)

    Parameters
    ----------
    encoder_kwargs : dict, optional
        Additional keyword arguments for scikit-learn TargetEncoder.
    random_state : int, optional
        Random state for reproducibility.

    Attributes
    ----------
    mappings_ : dict
        Per-feature mapping DataFrames with WOE info.
    encoders_ : dict
        Fitted TargetEncoder per feature.
    feature_stats_ : dict
        Per-feature statistics (Gini, IV, etc.).
    y_prior_ : float
        Mean target in fit data.
    """

    def __init__(self, encoder_kwargs=None, random_state=42):
        self.encoder_kwargs = encoder_kwargs or {}
        self.random_state = random_state
        self.encoders_ = {}
        self.mappings_ = {}
        self.feature_stats_ = {}
        self.y_prior_ = None
        self.is_fitted_ = False

    def _calculate_gini(self, y_true, y_pred):
        """Calculate Gini coefficient from AUC."""
        try:
            auc = roc_auc_score(y_true, y_pred)
            return 2 * auc - 1
        except ValueError:
            return np.nan  # Handle cases with single class

    def _calculate_iv(self, mapping_df, total_good, total_bad):
        """Calculate Information Value for a feature."""
        iv = 0
        for _, row in mapping_df.iterrows():
            bad_rate = (row["count"] * row["event_rate"]) / total_bad if total_bad > 0 else 0
            good_rate = (row["count"] * (1 - row["event_rate"])) / total_good if total_good > 0 else 0

            if good_rate > 0 and bad_rate > 0:
                iv += (bad_rate - good_rate) * row["woe"]
        return iv

    def _calculate_feature_stats(self, col, X, y, mapping_df):
        """Calculate comprehensive statistics for a feature."""

        # Basic counts
        total_obs = len(y)
        total_bad = y.sum()
        total_good = total_obs - total_bad

        # Get WOE transformed values for this feature
        woe_values = self.transform(X[[col]])[col]  # Use original column name

        return {
            "feature": col,
            "n_categories": len(mapping_df),
            "total_observations": total_obs,
            "missing_count": X[col].isna().sum(),
            "missing_rate": X[col].isna().mean(),
            "gini": self._calculate_gini(y, woe_values),
            "information_value": self._calculate_iv(mapping_df, total_good, total_bad),
            "min_woe": mapping_df["woe"].min(),
            "max_woe": mapping_df["woe"].max(),
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.y_prior_ = y.mean()
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        self.encoders_ = {}
        self.mappings_ = {}
        self.feature_stats_ = {}

        for col in X.columns:
            enc = TargetEncoder(**self.encoder_kwargs, random_state=self.random_state)
            enc.fit(X[[col]], y)
            self.encoders_[col] = enc

            # Get unique categories as seen by the encoder
            categories = enc.categories_[0]
            event_rates = enc.encodings_[0]

            # Defensive clipping to avoid log(0)
            event_rates = np.clip(event_rates, 1e-15, 1 - 1e-15)
            odds_cat = event_rates / (1 - event_rates)
            woe = np.log(odds_cat / odds_prior)

            # Count for each category in training data
            value_counts = X[col].value_counts(dropna=False)
            # Map in same order as categories_
            count = pd.Series(value_counts).reindex(categories).fillna(0).astype(int).values

            # Enhanced mapping with more details
            mapping_df = pd.DataFrame({
                "category": categories,
                "count": count,
                "count_pct": count / len(X) * 100,
                "event_rate": event_rates,
                "woe": woe,
                "good_count": (count * (1 - event_rates)).astype(int),
                "bad_count": (count * event_rates).astype(int),
            }).set_index("category")

            self.mappings_[col] = mapping_df

            # Calculate feature-level statistics
            self.feature_stats_[col] = self._calculate_feature_stats(col, X, y, mapping_df)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to WOE values, preserving original column names."""
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        woe_df = pd.DataFrame(index=X.index)

        for col in X.columns:
            enc = self.encoders_[col]
            event_rate = enc.transform(X[[col]])
            # scikit-learn returns np.ndarray
            if isinstance(event_rate, pd.DataFrame):
                event_rate = event_rate.values.flatten()
            event_rate = np.clip(event_rate, 1e-15, 1 - 1e-15)
            odds_cat = event_rate / (1 - event_rate)
            woe = np.log(odds_cat / odds_prior)
            woe_df[col] = woe  # Keep original column name

        return woe_df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the encoder and transform the data."""
        self.fit(X, y)
        return self.transform(X)

    def get_mapping(self, col: str) -> pd.DataFrame:
        """Return the mapping table for a feature (category, count, event_rate, woe)."""
        return self.mappings_[col].reset_index()

    def get_all_mappings(self) -> dict:
        """Get all mappings (useful for serialization, audit, or compact storage)."""
        return {col: mapping.reset_index() for col, mapping in self.mappings_.items()}

    def get_feature_stats(self, col: Optional[str] = None) -> pd.DataFrame:
        """Get feature statistics. If col is None, return stats for all features."""
        if col is not None:
            return pd.DataFrame([self.feature_stats_[col]])
        else:
            return pd.DataFrame(list(self.feature_stats_.values()))

    def get_feature_summary(self) -> pd.DataFrame:
        """Get a summary table of all features ranked by predictive power."""
        stats_df = self.get_feature_stats()
        return stats_df.sort_values("gini", ascending=False)[
            ["feature", "gini", "information_value", "n_categories"]
        ].round(4)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using WOE-transformed features.
        This is a simple linear combination - for real scoring you'd use logistic regression.
        """
        X_woe = self.transform(X)
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        woe_score = X_woe.sum(axis=1) + np.log(odds_prior)

        # Convert to probability (simple sigmoid transformation)
        prob = sigmoid(woe_score)
        return np.column_stack([1 - prob, prob])

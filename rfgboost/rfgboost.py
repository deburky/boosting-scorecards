"""rfgboost.py."""

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import TargetEncoder


class RFGBoost:
    """Random Forest Gradient Boosting (RFGBoost) model."""

    def __init__(
        self,
        n_estimators=10,
        rf_params=None,
        learning_rate=0.1,
        task="regression",
        cat_features=None,
    ):
        """RFGBoost constructor."""

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.rf_params = rf_params if rf_params is not None else {}
        self.task = task
        self.models = []
        self.initial_pred = None
        self.cat_features = cat_features
        self.woe_encoders = {}
        self.prior = None
        self.label_mapping = None
        self.feature_names_ = None

    def _ensure_numeric(self, y):
        """
        Ensure target values are numeric.

        Parameters:
        -----------
        y : array-like
            The target values.

        Returns:
        --------
        y_numeric : array-like
            Numeric representation of target values.
        """
        y = np.asarray(y)
        if y.dtype.kind in "OSU":  # Object, String, or Unicode
            unique_vals = np.unique(y)
            if len(unique_vals) == 2:  # Binary classification
                self.label_mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                return np.array([self.label_mapping[val] for val in y])
        return y

    def _woe_encode(self, X, y=None, fit=False):
        """
        Weight of Evidence encoding for categorical features.

        Parameters:
        -----------
        X : DataFrame or ndarray
            The input samples.
        y : array-like, optional
            The target values. Only needed when fit=True.
        fit : bool, default=False
            Whether to fit the encoders or just transform.

        Returns:
        --------
        X_encoded : ndarray
            The transformed data.
        """
        if self.cat_features is None or len(self.cat_features) == 0:
            return np.array(X)

        # Ensure X is a DataFrame for easier handling of categorical features
        if not isinstance(X, pd.DataFrame):
            # If X is not a DataFrame, convert it to one
            if hasattr(X, "columns"):  # If it has column names (e.g., pandas-like)
                X = pd.DataFrame(X, columns=X.columns)
            else:
                # A numpy array without column names
                all_features = list(range(X.shape[1]))
                X = pd.DataFrame(X, columns=all_features)

        X_encoded = X.copy()
        self.feature_names_ = (
            list(X_encoded.columns)
            if isinstance(X_encoded, pd.DataFrame)
            else list(range(X_encoded.shape[1]))
        )

        if fit:
            # Ensure y is numeric before calculating mean
            y_numeric = self._ensure_numeric(y)
            self.prior = np.mean(y_numeric)

            for col in self.cat_features:
                # Make sure column exists in DataFrame
                if col not in X_encoded.columns:
                    print(
                        f"Warning: Column {col} not found in data. Available columns: {X_encoded.columns}"
                    )
                    continue

                # Initialize and fit a target encoder
                te = TargetEncoder()
                te.fit(X_encoded[[col]], y_numeric)
                self.woe_encoders[col] = te

                # Transform the data
                te_encoded = te.transform(X_encoded[[col]]).flatten()

                # Calculate WOE: log((te_ / prior) / ((1 - te_) / (1 - prior)))
                # Avoiding division by zero
                eps = 1e-10
                te_encoded = np.clip(te_encoded, eps, 1 - eps)
                prior_safe = np.clip(self.prior, eps, 1 - eps)

                woe = np.log(
                    (te_encoded / prior_safe) / ((1 - te_encoded) / (1 - prior_safe))
                )
                X_encoded[col] = woe
        else:
            for col in self.cat_features:
                if col not in X_encoded.columns:
                    print(f"Warning: Column {col} not found in data")
                    continue

                if col in self.woe_encoders:
                    # Transform with the fitted encoder
                    te_encoded = (
                        self.woe_encoders[col].transform(X_encoded[[col]]).flatten()
                    )

                    # Calculate WOE using the stored prior
                    eps = 1e-10
                    te_encoded = np.clip(te_encoded, eps, 1 - eps)
                    prior_safe = np.clip(self.prior, eps, 1 - eps)

                    woe = np.log(
                        (te_encoded / prior_safe)
                        / ((1 - te_encoded) / (1 - prior_safe))
                    )
                    X_encoded[col] = woe

        # Convert to numpy array at the end
        return X_encoded.values

    def inverse_woe_transform(self, X_woe, cat_columns=None):
        """
        Transform WOE values back to probabilities.

        Parameters:
        -----------
        X_woe : DataFrame or ndarray
            The WOE-encoded data.
        cat_columns : list, optional
            List of categorical columns to transform. Defaults to self.cat_features.

        Returns:
        --------
        X_proba : array-like
            The probability-encoded data.
        """
        if cat_columns is None:
            cat_columns = self.cat_features

        if cat_columns is None or len(cat_columns) == 0:
            return X_woe.copy()

        # Ensure X_woe is a DataFrame for easier handling
        if not isinstance(X_woe, pd.DataFrame):
            if hasattr(X_woe, "columns"):  # If it has column names
                X_woe = pd.DataFrame(X_woe, columns=X_woe.columns)
            else:
                # If it's just a numpy array without column names
                X_woe = pd.DataFrame(X_woe)

        X_proba = X_woe.copy()

        for col in cat_columns:
            if col not in X_proba.columns:
                continue

            # Extract WOE values
            woe_values = X_proba[col].values

            # Convert WOE back to probabilities
            exp_woe = np.exp(woe_values)
            p = self.prior * exp_woe / (1 - self.prior + self.prior * exp_woe)

            X_proba[col] = p

        return X_proba

    def fit(self, X, y):
        """
        Fit the RFGBoost model.
        Parameters:
        -----------
        X : DataFrame or ndarray
            The input samples.
        y : array-like
            The target values.
        Returns:
        --------
        self : object
            Returns self.
        """
        # Ensure y is numeric
        y_numeric = self._ensure_numeric(y)

        # Apply WOE encoding if categorical features are specified
        if self.cat_features is not None and len(self.cat_features) > 0:
            X_encoded = self._woe_encode(X, y_numeric, fit=True)
        else:
            X_encoded = np.array(X)

        n_samples = X_encoded.shape[0]

        # Initialize predictions
        if self.task == "regression":
            self.initial_pred = np.mean(y_numeric)
            pred = np.full(n_samples, self.initial_pred)
        elif self.task == "classification":
            self.initial_pred = np.log(
                np.mean(y_numeric) / (1 - np.mean(y_numeric))
            )  # Logit for binary classification
            pred = np.full(n_samples, self.initial_pred)

        self.models = []
        update = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            # Compute residuals
            if self.task == "regression":
                residuals = y_numeric - pred
                rf = RandomForestRegressor(**self.rf_params)
                rf.fit(X_encoded, residuals)
                update = rf.predict(X_encoded)
            elif self.task == "classification":
                p = 1 / (1 + np.exp(-pred))
                residuals = (y_numeric - p) / (
                    p * (1 - p)
                )  # Working response (FHT2000)
                rf = RandomForestRegressor(
                    **self.rf_params
                )  # Use regression for gradients
                rf.fit(X_encoded, residuals)
                update = rf.predict(X_encoded)

            self.models.append(rf)
            pred += self.learning_rate * update

        return self

    def predict(self, X):
        """
        Predict using the fitted model.
        Parameters:
        -----------
        X : DataFrame or ndarray
            The input samples.
        Returns:
        --------
        pred : array-like
            The predicted values.
        """
        # Apply WOE encoding for prediction
        if self.cat_features is not None and len(self.cat_features) > 0:
            X_encoded = self._woe_encode(X, fit=False)
        else:
            X_encoded = np.array(X)

        pred = np.full(X_encoded.shape[0], self.initial_pred)

        for rf in self.models:
            pred += self.learning_rate * rf.predict(X_encoded)

        return 1 / (1 + np.exp(-pred)) if self.task == "classification" else pred

    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted model.
        Parameters:
        -----------
        X : DataFrame or ndarray
            The input samples.
        Returns:
        --------
        proba : array-like
            The predicted class probabilities.
        """
        pred = self.predict(X)
        return (
            np.column_stack((1 - pred, pred)) if self.task == "classification" else pred
        )

    def get_feature_importance(self):
        """
        Extract feature importance from the fitted RFGBoost model.

        Parameters:
        -----------
        all_features : list
            List of all feature names (after encoding).

        Returns:
        --------
        feature_importance_df : DataFrame
            Feature names and their average importance across all internal RF models.
        """
        if not self.models:
            raise ValueError("The model has not been fitted yet.")

        feature_importance = np.zeros(len(self.feature_names_))

        for rf_model in self.models:
            feature_importance += rf_model.feature_importances_

        feature_importance /= len(self.models)

        feature_importance_df = pd.DataFrame(
            {"Feature": self.feature_names_, "Importance": feature_importance}
        )

        return feature_importance_df.sort_values(
            "Importance", ascending=False
        ).reset_index(drop=True)

    def predict_ci(self, X, alpha=0.05):
        """
        Compute confidence intervals for predicted logits or probabilities.

        Parameters:
            X (DataFrame or array): Input features.
            method (str): "logit" for CI on log-odds, "proba" for CI on probabilities.
            alpha (float): Significance level (default 0.05 for 95% CI).
        Returns:
            np.ndarray: [lower_bound, upper_bound] for each sample.
        """
        if self.cat_features:
            X_encoded = self._woe_encode(X, fit=False)
        else:
            X_encoded = np.array(X)

        # Predict mean and variance (std^2) using ensemble of random forests
        pred_mean = np.full(X_encoded.shape[0], self.initial_pred)
        pred_var = np.zeros(X_encoded.shape[0])

        for rf in self.models:
            tree_preds = np.array([tree.predict(X_encoded) for tree in rf.estimators_])
            tree_mean = tree_preds.mean(axis=0)
            tree_var = tree_preds.var(axis=0)
            pred_mean += self.learning_rate * tree_mean
            pred_var += (self.learning_rate**2) * tree_var

        z_crit = norm.ppf(1 - alpha / 2)
        std = np.sqrt(pred_var)

        if self.task == "classification":
            lower = sigmoid(pred_mean - z_crit * std)
            upper = sigmoid(pred_mean + z_crit * std)
        elif self.task == "regression":
            lower = pred_mean - z_crit * std
            upper = pred_mean + z_crit * std

        return np.vstack([lower, upper]).T

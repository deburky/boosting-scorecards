#!/usr/bin/env python3
"""
XGBoost: Aggregated Binomial (no expansion) vs Expanded Bernoulli

What this script does
---------------------
1) Generates synthetic *aggregated* binomial data with (X, success_rate, trials)
2) Trains XGBoost with a custom objective that fits the aggregated binomial NLL
3) Trains XGBoost on an *expanded* Bernoulli dataset for comparison
4) Compares validation metrics (binomial NLL, RMSE) and training time
5) Plots:
   - metric comparison (bar chart)
   - calibration curves (aggregated vs expanded)
   - predicted vs true scatter (two separate figures)

Notes
-----
- The "expanded" approach is only for small demos. Do NOT expand rows for huge datasets.
- Uses `xgb.train` (+ DMatrix) so the custom objective can access sample weights (trials).
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy.special import expit as sigmoid  # pylint: disable=no-name-in-module
from sklearn.metrics import mean_squared_error as brier_score_loss


def make_synthetic(n_rows=30000, n_features=15, trials_mean=10, seed=7):
    """
    Create aggregated-binomial data:
      X  (n_rows, n_features)
      y_rate = successes / trials   (float in [0,1])
      trials  (>=1 per row)
      p_true  underlying oracle probability used to generate outcomes
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)

    beta = rng.normal(size=(n_features,)).astype(np.float32)
    logit = X @ beta + rng.normal(scale=0.6, size=n_rows).astype(np.float32)
    p = sigmoid(logit)

    trials = rng.poisson(lam=trials_mean, size=n_rows).astype(np.int32) + 1
    successes = rng.binomial(n=trials, p=p).astype(np.int32)
    y_rate = successes / trials

    return X, y_rate.astype(np.float32), trials.astype(np.float32), p.astype(np.float32)


def expand_rows(X, y_rate, trials, seed=123):
    """
    Expand aggregated rows into Bernoulli observations:
    For i with n_i trials and rate y_i, create n_i rows with target in {0,1}.
    WARNING: This explodes row count; only for small demos.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    counts = trials.astype(int)
    total = int(np.sum(counts))

    X_exp = np.empty((total, X.shape[1]), dtype=X.dtype)
    y_exp = np.empty((total,), dtype=np.float32)

    start = 0
    for i in range(n):
        k = counts[i]
        end = start + k
        X_exp[start:end] = X[i]
        y_exp[start:end] = rng.binomial(1, p=float(y_rate[i]), size=k).astype(
            np.float32
        )
        start = end

    return X_exp, y_exp


def binomial_objective(preds, dtrain):
    """
    Custom objective for aggregated binomial with per-row trial weights n.
    y is the observed success rate (successes / n), preds are raw scores (logits).
      grad = n * (p - y)
      hess = n * p * (1 - p)
    """
    y = dtrain.get_label()
    n = dtrain.get_weight()
    p = sigmoid(preds)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    grad = n * (p - y)
    hess = n * p * (1.0 - p)
    return grad, hess


def binomial_metric(preds, dtrain):
    """
    Proper binomial negative log-likelihood per *trial* (lower is better).
    """
    y = dtrain.get_label()
    n = dtrain.get_weight()
    p = sigmoid(preds)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    k = y * n
    nll = -(k * np.log(p) + (n - k) * np.log(1 - p))
    # average per trial
    return "binom_nll", float(np.sum(nll) / np.sum(n))


def rmse_rate_metric(preds, dtrain):
    """
    RMSE between predicted probability and observed success rate.
    """
    y = dtrain.get_label()
    p = sigmoid(preds)
    return "rmse", float(np.sqrt(np.mean((p - y) ** 2)))


def binom_nll_per_row(p_hat, y_rate, trials):
    """Compute binomial NLL averaged per *trial* on aggregated rows."""
    p_hat = np.clip(p_hat, 1e-15, 1 - 1e-15)
    k = y_rate * trials
    nll = -(k * np.log(p_hat) + (trials - k) * np.log(1 - p_hat))
    return float(np.sum(nll) / np.sum(trials))


def rmse_per_row(p_hat, y_rate):
    """RMSE between predicted probability and observed success rate."""
    return float(np.sqrt(np.mean((p_hat - y_rate) ** 2)))


def calibration_curve(p_hat, y_rate, n_bins=20):
    """
    Bin predictions into equal-width bins, return (avg_pred, avg_true) per bin.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_hat, bins) - 1
    avg_pred, avg_true = [], []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        avg_pred.append(float(np.mean(p_hat[mask])))
        avg_true.append(float(np.mean(y_rate[mask])))
    return np.array(avg_pred), np.array(avg_true)


# ---------------------------
# Main
# ---------------------------
def main():  # sourcery skip: extract-duplicate-method
    """Main function."""
    # 1) Data
    X, y_rate, trials, p_true = make_synthetic()
    rng = np.random.default_rng(0)
    idx = rng.permutation(X.shape[0])
    cut = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:cut], idx[cut:]

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y_rate[tr_idx], y_rate[va_idx]
    w_tr, w_va = trials[tr_idx], trials[va_idx]
    p_true_va = p_true[va_idx]

    # 2) Aggregated training (custom objective)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va, weight=w_va)

    params = {
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "max_bin": 256,
        "lambda": 1.0,
        "verbosity": 1,
    }

    t0 = time.time()
    bst_agg = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        obj=binomial_objective,
        feval=lambda preds, d: [binomial_metric(preds, d), rmse_rate_metric(preds, d)],
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
    )
    t_agg = time.time() - t0

    logit_va_agg = bst_agg.predict(dvalid, output_margin=True)
    p_va_agg = sigmoid(logit_va_agg)

    # 3) Expanded Bernoulli baseline (for comparison on small data only)
    X_tr_exp, y_tr_exp = expand_rows(X_tr, y_tr, w_tr, seed=11)
    X_va_exp, y_va_exp = expand_rows(X_va, y_va, w_va, seed=22)

    dtrain_exp = xgb.DMatrix(X_tr_exp, label=y_tr_exp)
    dvalid_exp = xgb.DMatrix(X_va_exp, label=y_va_exp)

    params_exp = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "max_bin": 256,
        "lambda": 1.0,
        "verbosity": 1,
    }

    t1 = time.time()
    bst_exp = xgb.train(
        params=params_exp,
        dtrain=dtrain_exp,
        num_boost_round=500,
        evals=[(dtrain_exp, "train"), (dvalid_exp, "valid")],
        early_stopping_rounds=50,
    )
    t_exp = time.time() - t1

    # Predict once per original row for expanded model
    dvalid_rows = xgb.DMatrix(X_va)
    p_va_exp = bst_exp.predict(dvalid_rows)

    # 4) Metrics
    nll_agg = binom_nll_per_row(p_va_agg, y_va, w_va)
    nll_exp = binom_nll_per_row(p_va_exp, y_va, w_va)
    rmse_agg = rmse_per_row(p_va_agg, y_va)
    rmse_exp = rmse_per_row(p_va_exp, y_va)

    print("=== Validation Summary ===")
    print(
        f"Aggregated (custom obj): binom_nll={nll_agg:.6f}, rmse={rmse_agg:.6f}, train_time_s={t_agg:.2f}, rows_train={X_tr.shape[0]}, rows_valid={X_va.shape[0]}"
    )
    print(
        f"Expanded Bernoulli     : binom_nll={nll_exp:.6f}, rmse={rmse_exp:.6f}, train_time_s={t_exp:.2f}, rows_train={X_tr_exp.shape[0]}, rows_valid={X_va_exp.shape[0]}"
    )

    # 5) Plots (one figure per plot; no seaborn; no styles)
    # a) Metric comparison bar chart
    metrics = ["Binom NLL", "RMSE", "Train time (s)"]
    agg_vals = [nll_agg, rmse_agg, t_agg]
    exp_vals = [nll_exp, rmse_exp, t_exp]
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, agg_vals, width, label="Aggregated")
    plt.bar(x + width / 2, exp_vals, width, label="Expanded")
    plt.xticks(x, metrics)
    plt.legend()
    plt.title("Validation metrics and train time")
    plt.tight_layout()
    plt.show()

    # b) Calibration curves (bin avg predicted vs observed)
    avg_pred_agg, avg_true_agg = calibration_curve(p_va_agg, y_va, n_bins=20)
    avg_pred_exp, avg_true_exp = calibration_curve(p_va_exp, y_va, n_bins=20)

    # Print brier scores
    print(f"Brier score (Aggregated): {brier_score_loss(y_va, p_va_agg)}")
    print(f"Brier score (Expanded): {brier_score_loss(y_va, p_va_exp)}")

    print(f"Target (Aggregated): {y_va[:10]}")
    print(f"Target (Expanded): {y_va_exp[:10]}")

    plt.figure()
    plt.plot(avg_pred_agg, avg_true_agg, marker="o", linestyle="-", label="Aggregated")
    plt.plot(avg_pred_exp, avg_true_exp, marker="o", linestyle="-", label="Expanded")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability (bin avg)")
    plt.ylabel("Observed success rate (bin avg)")
    plt.title("Calibration curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # c) Predicted vs true scatter (aggregated)
    plt.figure()
    plt.scatter(p_true_va, p_va_agg, s=5, alpha=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("True p (oracle)")
    plt.ylabel("Predicted p")
    plt.title("Predicted vs True p – Aggregated")
    plt.tight_layout()
    plt.show()

    # d) Predicted vs true scatter (expanded)
    plt.figure()
    plt.scatter(p_true_va, p_va_exp, s=5, alpha=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("True p (oracle)")
    plt.ylabel("Predicted p")
    plt.title("Predicted vs True p – Expanded")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

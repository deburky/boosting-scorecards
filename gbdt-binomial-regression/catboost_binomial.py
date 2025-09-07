#!/usr/bin/env python3
"""
CatBoost: Aggregated Binomial via weighted CrossEntropy vs Expanded Bernoulli (baseline)

What this script does
---------------------
1) Generates synthetic *aggregated* binomial data with (X, success_rate, trials).
2) Trains CatBoost to match the aggregated binomial likelihood WITHOUT expansion:
    - Train with label=y_rate in [0,1] and weight=trials using loss_function="CrossEntropy".
    - This is equivalent to fitting the binomial NLL per trial (up to a constant).
3) Trains CatBoost on a fully *expanded* Bernoulli dataset for comparison (one row per trial).
4) Compares validation metrics (binom NLL, RMSE, Brier) and training time.
5) Plots:
   - metric comparison (bar chart)
   - calibration curves (aggregated vs expanded)
   - predicted vs true scatter (two separate figures)

Notes
-----
- Do NOT fully expand rows for large datasets; it's only for small demos.
- CrossEntropy accepts soft labels in [0,1]; with per-row weights=trials the objective
  equals the aggregated binomial NLL up to an additive constant.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.special import expit as sigmoid  # pylint: disable=no-name-in-module


# ---------------------------
# Data helpers
# ---------------------------
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


# ---------------------------
# Metrics
# ---------------------------
def binom_nll_per_row(p_hat, y_rate, trials):
    """Binomial negative log-likelihood averaged per *trial* on aggregated rows."""
    p_hat = np.clip(p_hat, 1e-15, 1 - 1e-15)
    k = y_rate * trials
    nll = -(k * np.log(p_hat) + (trials - k) * np.log(1 - p_hat))
    return float(np.sum(nll) / np.sum(trials))


def rmse_per_row(p_hat, y_rate):
    """RMSE between predicted probability and observed success rate."""
    return float(np.sqrt(np.mean((p_hat - y_rate) ** 2)))


def brier_score(y_rate, p_hat):
    """Brier score (mean squared error for probabilities)."""
    return float(np.mean((p_hat - y_rate) ** 2))


def calibration_curve(p_hat, y_rate, n_bins=20):
    """Bin predictions into equal-width bins; return (avg_pred, avg_true) per bin."""
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

    # Train with soft labels y in [0,1] and per-row weights=trials using CrossEntropy.
    train_pool_agg = Pool(X_tr, label=y_tr, weight=w_tr)
    valid_pool_agg = Pool(X_va, label=y_va, weight=w_va)

    params_agg = dict(
        loss_function="CrossEntropy",  # accepts soft labels in [0,1]
        eval_metric="CrossEntropy",
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=0,
        verbose=False,
        od_type="Iter",
        od_wait=50,
        use_best_model=True,
    )

    t0 = time.time()
    model_agg = CatBoostClassifier(**params_agg)
    model_agg.fit(train_pool_agg, eval_set=valid_pool_agg)
    t_agg = time.time() - t0

    # Predict probability for the original validation rows
    p_va_agg = model_agg.predict_proba(X_va)[:, 1]

    # 3) Expanded Bernoulli baseline (one row per trial) — for comparison only
    X_tr_exp, y_tr_exp = expand_rows(X_tr, y_tr, w_tr, seed=11)
    X_va_exp, y_va_exp = expand_rows(X_va, y_va, w_va, seed=22)

    train_pool_exp = Pool(X_tr_exp, label=y_tr_exp)
    valid_pool_exp = Pool(X_va_exp, label=y_va_exp)
    valid_pool_exp_rows = Pool(X_va)

    params_exp = dict(
        loss_function="Logloss",  # hard 0/1 labels on expanded data
        eval_metric="Logloss",
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=0,
        verbose=False,
        od_type="Iter",
        od_wait=50,
        use_best_model=True,
    )

    t1 = time.time()
    model_exp = CatBoostClassifier(**params_exp)
    model_exp.fit(train_pool_exp, eval_set=valid_pool_exp)
    t_exp = time.time() - t1

    # Predict once per original validation row (features identical per expanded row)
    p_va_exp = model_exp.predict_proba(valid_pool_exp_rows)[:, 1]

    # 4) Metrics on aggregated validation rows
    nll_agg = binom_nll_per_row(p_va_agg, y_va, w_va)
    nll_exp = binom_nll_per_row(p_va_exp, y_va, w_va)
    rmse_agg = rmse_per_row(p_va_agg, y_va)
    rmse_exp = rmse_per_row(p_va_exp, y_va)
    brier_agg = brier_score(y_va, p_va_agg)
    brier_exp = brier_score(y_va, p_va_exp)

    print("Validation Summary (CatBoost)")
    print(
        f"Aggregated (CrossEntropy+weights): binom_nll={nll_agg:.6f}, rmse={rmse_agg:.6f}, "
        f"brier={brier_agg:.6f}, train_time_s={t_agg:.2f}, rows_train={X_tr.shape[0]}, "
        f"rows_valid={X_va.shape[0]}"
    )
    print(
        f"Expanded Bernoulli: binom_nll={nll_exp:.6f}, rmse={rmse_exp:.6f}, "
        f"brier={brier_exp:.6f}, train_time_s={t_exp:.2f}, rows_train={X_tr_exp.shape[0]}, "
        f"rows_valid={X_va_exp.shape[0]}"
    )

    # 5) Plots (no seaborn; one chart per figure)
    # a) Metric comparison
    metrics = ["Binom NLL", "RMSE", "Brier", "Train time (s)"]
    agg_vals = [nll_agg, rmse_agg, brier_agg, t_agg]
    exp_vals = [nll_exp, rmse_exp, brier_exp, t_exp]
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, agg_vals, width, label="Aggregated")
    plt.bar(x + width / 2, exp_vals, width, label="Expanded")
    plt.xticks(x, metrics)
    plt.legend()
    plt.title("Validation metrics and train time (CatBoost)")
    plt.tight_layout()
    plt.show()

    # b) Calibration curves
    avg_pred_agg, avg_true_agg = calibration_curve(p_va_agg, y_va, n_bins=20)
    avg_pred_exp, avg_true_exp = calibration_curve(p_va_exp, y_va, n_bins=20)

    plt.figure()
    plt.plot(avg_pred_agg, avg_true_agg, marker="o", linestyle="-", label="Aggregated")
    plt.plot(avg_pred_exp, avg_true_exp, marker="o", linestyle="-", label="Expanded")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability (bin avg)")
    plt.ylabel("Observed success rate (bin avg)")
    plt.title("Calibration curves (CatBoost)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # c) Predicted vs true scatter (aggregated)
    plt.figure()
    plt.scatter(p_true_va, p_va_agg, s=5, alpha=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("True p (oracle)")
    plt.ylabel("Predicted p")
    plt.title("Predicted vs True p - Aggregated (CatBoost)")
    plt.tight_layout()
    plt.show()

    # d) Predicted vs true scatter (expanded)
    plt.figure()
    plt.scatter(p_true_va, p_va_exp, s=5, alpha=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("True p (oracle)")
    plt.ylabel("Predicted p")
    plt.title("Predicted vs True p - Expanded (CatBoost)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CatBoost: Aggregated Binomial (no full expansion) vs Expanded Bernoulli

What this script does
---------------------
1) Generates synthetic *aggregated* binomial data with (X, success_rate, trials).
2) Trains CatBoost in a way that matches the aggregated binomial likelihood
   WITHOUT expanding to one row per trial:
      - For each original row i with k_i successes out of n_i trials,
        we create TWO training rows that share the same features:
           - one with label=1 and weight=k_i
           - one with label=0 and weight=n_i - k_i
      - This is equivalent to fitting the binomial NLL and only doubles the row count.
3) Trains CatBoost on a fully *expanded* Bernoulli dataset for comparison (one row per trial).
4) Compares validation metrics (binom NLL, RMSE, Brier) and training time.
5) Plots:
   - metric comparison (bar chart)
   - calibration curves (aggregated vs expanded)
   - predicted vs true scatter (two separate figures)

Notes
-----
- CatBoost does not support a custom objective for aggregated-binomial directly.
  The 2-rows-per-original trick (success/failure with weights) is the correct way.
- The expanded approach explodes rows by ~sum(trials) and is shown only for small demos.
"""

import time

import numpy as np
from catboost import CatBoostClassifier, Pool
from plotting_utils import (
    create_combined_plot,
    print_brier_scores,
    print_target_samples,
)
from scipy.special import expit as sigmoid  # pylint: disable=no-name-in-module
from sklearn.metrics import mean_squared_error as brier_score_loss


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


def expand_rows(X, y_rate, trials):
    """
    Expand aggregated rows into Bernoulli observations:
    For i with n_i trials and rate y_i, create n_i rows with target in {0,1}.
    WARNING: This explodes row count; only for small demos.
    """
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
        successes = int(round(y_rate[i] * k))  # recover successes
        y_exp[start : start + successes] = 1.0
        y_exp[start + successes : end] = 0.0
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


# ---------------------------
# Main
# ---------------------------
def main():
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

    # 2) CatBoost "aggregated" via 2-rows-per-original (success/failure with weights)
    # Construct training set:
    #   row i => two duplicates of X[i]:
    #     label=1 weight=successes_i, label=0 weight=(trials_i - successes_i)
    successes_tr = (y_tr * w_tr).astype(np.float32)
    fails_tr = w_tr - successes_tr

    X_pos = X_tr.copy()
    y_pos = np.ones_like(y_tr, dtype=np.float32)
    w_pos = successes_tr

    X_neg = X_tr.copy()
    y_neg = np.zeros_like(y_tr, dtype=np.float32)
    w_neg = fails_tr

    X_agg = np.concatenate([X_pos, X_neg], axis=0)
    y_agg = np.concatenate([y_pos, y_neg], axis=0)
    w_agg = np.concatenate([w_pos, w_neg], axis=0)

    print(f"X_pos: {X_pos.shape}")
    print(f"y_pos: {y_pos.shape}")
    print(f"w_pos: {w_pos.shape}")
    print("--------------------------------")
    print(f"X_agg: {X_agg.shape}")
    print(f"y_agg: {y_agg.shape}")
    print(f"w_agg: {w_agg.shape}")

    # Validation set: we evaluate on original rows, so we just use X_va and y_va/w_va
    train_pool_agg = Pool(X_agg, label=y_agg, weight=w_agg)
    valid_pool_rows = Pool(X_va, label=None)  # for getting per-row probabilities

    # Train CatBoost (classification with Logloss)
    params_agg = dict(
        loss_function="Logloss",
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=0,
        verbose=False,
    )

    t0 = time.time()
    model_agg = CatBoostClassifier(**params_agg)
    model_agg.fit(train_pool_agg)
    t_agg = time.time() - t0

    # Predict probability for the original validation rows
    p_va_agg = model_agg.predict_proba(valid_pool_rows)[:, 1]

    # 3) Expanded Bernoulli baseline (one row per trial) — for comparison only
    X_tr_exp, y_tr_exp = expand_rows(X_tr, y_tr, w_tr)
    X_va_exp, y_va_exp = expand_rows(X_va, y_va, w_va)

    train_pool_exp = Pool(X_tr_exp, label=y_tr_exp)
    valid_pool_exp_rows = Pool(X_va)  # we'll predict once per original row

    params_exp = dict(
        loss_function="Logloss",
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=0,
        verbose=False,
    )

    t1 = time.time()
    model_exp = CatBoostClassifier(**params_exp)
    model_exp.fit(train_pool_exp)
    t_exp = time.time() - t1

    # Predict once per original validation row (features are identical per expanded row)
    p_va_exp = model_exp.predict_proba(valid_pool_exp_rows)[:, 1]

    print(f"Target (Aggregated): {y_agg[:10]}")
    print(f"Target (Expanded): {y_va_exp[:10]}")

    # 4) Metrics on aggregated validation rows
    nll_agg = binom_nll_per_row(p_va_agg, y_va, w_va)
    nll_exp = binom_nll_per_row(p_va_exp, y_va, w_va)
    rmse_agg = rmse_per_row(p_va_agg, y_va)
    rmse_exp = rmse_per_row(p_va_exp, y_va)
    brier_agg = brier_score_loss(y_va, p_va_agg)
    brier_exp = brier_score_loss(y_va, p_va_exp)

    print("=== Validation Summary (CatBoost) ===")
    print(
        f"Aggregated (2-rows trick): binom_nll={nll_agg:.6f}, rmse={rmse_agg:.6f}, "
        f"brier={brier_agg:.6f}, train_time_s={t_agg:.2f}, rows_train={X_agg.shape[0]}, rows_valid={X_va.shape[0]}"
    )
    print(
        f"Expanded Bernoulli      : binom_nll={nll_exp:.6f}, rmse={rmse_exp:.6f}, "
        f"brier={brier_exp:.6f}, train_time_s={t_exp:.2f}, rows_train={X_tr_exp.shape[0]}, rows_valid={X_va_exp.shape[0]}"
    )

    # Print brier scores and target samples
    print_brier_scores(y_va, p_va_agg, p_va_exp, "Aggregated", "Expanded")
    print_target_samples(y_va, y_va_exp, "Aggregated", "Expanded")

    # Create combined plot using shared plotting utilities
    create_combined_plot(
        p_va_agg=p_va_agg,
        p_va_exp=p_va_exp,
        y_va=y_va,
        nll_agg=nll_agg,
        nll_exp=nll_exp,
        rmse_agg=rmse_agg,
        rmse_exp=rmse_exp,
        t_agg=t_agg,
        t_exp=t_exp,
        filename="catboost_binomial_two_trick.png",
        agg_label="Binomial",
        exp_label="Log Loss",
    )


if __name__ == "__main__":
    main()

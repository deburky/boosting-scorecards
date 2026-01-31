#!/usr/bin/env python3
"""Gini, Risk Appetite, and Economic Gains in Credit Risk.

This script generates two visualizations:
1. CAP curve + Profit vs Approval Rate (profit optimization)
2. CAP curve + Approval Rate vs Bad Rate (risk appetite/cutoff selection)

A common misconception is that Gini has no relationship to business metrics.
Gini quantifies model power in separating good risk from bad risk. It is
related to AUC (concordance probability) as:

Gini = P(concordant) - P(discordant) = AUC - (1 - AUC) = 2 * AUC - 1

This script demonstrates how Gini translates to economic value.

Assumptions:
- Loan unit economics: loan_size * [(1 - Bad rate) * Margin - Bad rate]
- Risk appetite: accept only 5% bad rate in the approved book

At a fixed target bad rate (5%), a higher-Gini model approves more applicants
while meeting the same risk; the economic gain is the extra volume at the
same unit profit per loan.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

np.random.seed(42)

# Population and risk parameters
N = 10_000
TRUE_BAD_RATE = 0.1
TARGET_BAD_RATE = 0.05  # Risk appetite: accept only 5% bad in approved book

# Unit economics: profit per loan = L * [(1 - p) * MARGIN - p] where p = default prob
LOAN_SIZE = 10_000
MARGIN = 0.12

# Plotting style
plt.rcParams["font.family"] = "Arial"
COLORS = ["#7fc1ff", "#d291ff"]  # Model A, Model B
IDEAL_COLOR = "#ffb870"
RANDOM_COLOR = "#222222"
RED_COLOR = "#e16e77"

# Generate population
true_labels = np.random.binomial(1, TRUE_BAD_RATE, N)


def _d_from_auc(desired_auc: float) -> float:
    """Calculate separation d for N(0,1) vs N(d,1) distributions.

    Args:
        desired_auc: Target AUC value.

    Returns:
        Separation parameter d that achieves the desired AUC.
    """
    t = np.sqrt(np.log(1 / (1 - desired_auc) ** 2))
    z = t - (
        (2.515517 + 0.802853 * t + 0.0103328 * t**2)
        / (1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3)
    )
    return z * np.sqrt(2)


def generate_scores_for_gini(
    labels: np.ndarray, target_gini: float, st_dev: float = 0.30
) -> np.ndarray:
    """Generate scores that achieve approximately target Gini coefficient.

    Args:
        labels: Binary labels (0/1) for the population.
        target_gini: Target Gini coefficient (e.g., 0.4 for 40% Gini).
        st_dev: Standard deviation for score distributions.

    Returns:
        Array of probability scores between 0 and 1.
    """
    desired_auc = (target_gini + 1) / 2
    d_unit = _d_from_auc(desired_auc)
    d = d_unit * st_dev

    n_event = int(labels.sum())
    n_non_event = len(labels) - n_event
    x_event = np.random.normal(d, st_dev, n_event)
    x_non_event = np.random.normal(0, st_dev, n_non_event)

    x = np.empty(len(labels))
    x[labels == 0] = x_non_event
    x[labels == 1] = x_event

    desired_mean = -2.0
    desired_std = 0.399522
    x = (x - np.mean(x)) * (desired_std / np.std(x)) + desired_mean
    x_min = np.min(x)
    x_range = np.max(x) - x_min
    if x_range > 0:
        desired_min = -2.935578
        desired_max = -1.173120
        x = (x - x_min) * (desired_max - desired_min) / x_range + desired_min
    return 1 / (1 + np.exp(-x))


def calibrate_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calibrate probability scores using logistic regression.

    Args:
        scores: Uncalibrated probability scores.
        labels: True binary labels.

    Returns:
        Calibrated probability scores.
    """
    lr = LogisticRegression(penalty=None)
    logit_scores = np.log(scores / (1 - scores))
    lr.fit(logit_scores.reshape(-1, 1), labels)
    return lr.predict_proba(logit_scores.reshape(-1, 1))[:, 1]


# Generate model scores with target Gini coefficients
scores_weak_raw = generate_scores_for_gini(true_labels, 0.4)
scores_strong_raw = generate_scores_for_gini(true_labels, 0.5)

scores_weak = calibrate_scores(scores_weak_raw, true_labels)
scores_strong = calibrate_scores(scores_strong_raw, true_labels)


def calculate_cap_curve(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Cumulative Accuracy Profile (CAP) curve coordinates.

    Args:
        scores: Probability scores for each observation.
        labels: True binary labels.

    Returns:
        Tuple of (population_fraction, recall, sort_idx) where:
            - population_fraction: Fraction of population (x-axis)
            - recall: Fraction of bad cases captured (y-axis)
            - sort_idx: Indices that sort scores in descending order
    """
    sort_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[sort_idx]
    n = len(labels)
    n_bads = labels.sum()
    population_fraction = np.arange(1, n + 1) / n
    cumsum_bads = np.cumsum(sorted_labels)
    recall = cumsum_bads / n_bads
    return population_fraction, recall, sort_idx


# Calculate CAP curves and Gini coefficients
pop_frac_weak, recall_weak, sort_idx_weak = calculate_cap_curve(
    scores_weak, true_labels
)
pop_frac_strong, recall_strong, sort_idx_strong = calculate_cap_curve(
    scores_strong, true_labels
)

auc_weak = roc_auc_score(true_labels, scores_weak)
auc_strong = roc_auc_score(true_labels, scores_strong)
gini_weak = 2 * auc_weak - 1
gini_strong = 2 * auc_strong - 1


def profit_curve_by_reject_count(
    scores: np.ndarray, labels: np.ndarray, loan_size: float, margin: float
) -> Tuple[list, list]:
    """Calculate profit curve for different approval rates.

    Accepts best k loans by score (lowest score = best).
    Unit economics: profit = loan_size * ((1 - score) * margin - score)

    Args:
        scores: Probability scores for each observation.
        labels: True binary labels.
        loan_size: Size of each loan.
        margin: Profit margin on good loans.

    Returns:
        Tuple of (approval_rates, profits) lists.
    """
    sort_idx = np.argsort(scores)  # Best (lowest score) first
    sorted_scores = scores[sort_idx]
    n = len(scores)
    approval_rates = []
    profits = []

    for k in range(n + 1):
        if k == 0:
            approval_rates.append(0.0)
            profits.append(0.0)
            continue

        approved_scores = sorted_scores[:k]
        individual_profits = loan_size * (
            (1 - approved_scores) * margin - approved_scores
        )
        total_profit = individual_profits.sum()

        approval_rates.append(k / n)
        profits.append(total_profit)

    return approval_rates, profits


def find_cutoff_at_target_bad_rate(
    scores: np.ndarray, labels: np.ndarray, target_bad_rate: float
) -> Tuple[float, float, int]:
    """Find optimal approval cutoff to achieve target bad rate.

    Args:
        scores: Probability scores for each observation.
        labels: True binary labels.
        target_bad_rate: Desired bad rate in approved population.

    Returns:
        Tuple of (approval_rate, achieved_bad_rate, best_reject_count).
    """
    sort_idx = np.argsort(scores)  # Ascending: best (lowest score) first
    sorted_labels = labels[sort_idx]

    n_total = len(sorted_labels)
    best_reject_count = 0
    best_diff = float("inf")

    for reject_count in range(n_total):
        approved_labels = sorted_labels[: n_total - reject_count]
        if len(approved_labels) == 0:
            continue

        approved_bad_rate = approved_labels.sum() / len(approved_labels)
        diff = abs(approved_bad_rate - target_bad_rate)

        if diff < best_diff:
            best_diff = diff
            best_reject_count = reject_count

    approval_rate = (n_total - best_reject_count) / n_total
    approved_labels = sorted_labels[: n_total - best_reject_count]
    achieved_bad_rate = (
        approved_labels.sum() / len(approved_labels) if len(approved_labels) > 0 else 0
    )

    return approval_rate, achieved_bad_rate, best_reject_count


approval_curve_weak, profit_curve_weak = profit_curve_by_reject_count(
    scores_weak, true_labels, LOAN_SIZE, MARGIN
)
approval_curve_strong, profit_curve_strong = profit_curve_by_reject_count(
    scores_strong, true_labels, LOAN_SIZE, MARGIN
)

# Find optimal profit points
max_profit_weak_idx = profit_curve_weak.index(max(profit_curve_weak))
max_profit_strong_idx = profit_curve_strong.index(max(profit_curve_strong))

approval_opt_weak = approval_curve_weak[max_profit_weak_idx]
profit_opt_weak = profit_curve_weak[max_profit_weak_idx]
approval_opt_strong = approval_curve_strong[max_profit_strong_idx]
profit_opt_strong = profit_curve_strong[max_profit_strong_idx]

# Calculate realized bad rates at optimal points
sort_idx_weak = np.argsort(scores_weak)  # Ascending: best first
sort_idx_strong = np.argsort(scores_strong)
sorted_labels_weak = true_labels[sort_idx_weak]
sorted_labels_strong = true_labels[sort_idx_strong]

n_approved_weak = int(approval_opt_weak * N)
n_approved_strong = int(approval_opt_strong * N)

bad_rate_opt_weak = (
    sorted_labels_weak[:n_approved_weak].mean() if n_approved_weak > 0 else 0
)
bad_rate_opt_strong = (
    sorted_labels_strong[:n_approved_strong].mean() if n_approved_strong > 0 else 0
)

# Calculate gains from stronger model
extra_approval_pct = (approval_opt_strong - approval_opt_weak) * 100
extra_profit = profit_opt_strong - profit_opt_weak
extra_profit_pct = 100 * extra_profit / profit_opt_weak if profit_opt_weak > 0 else 0

# Find cutoff points at target bad rate
approval_weak_cutoff, bad_rate_weak_cutoff, reject_count_weak = (
    find_cutoff_at_target_bad_rate(scores_weak, true_labels, TARGET_BAD_RATE)
)
approval_strong_cutoff, bad_rate_strong_cutoff, reject_count_strong = (
    find_cutoff_at_target_bad_rate(scores_strong, true_labels, TARGET_BAD_RATE)
)

# Calculate volume and economic gains at target bad rate
n_approved_weak_cutoff = N - reject_count_weak
n_approved_strong_cutoff = N - reject_count_strong

# Calculate bad rate curves for approval vs bad rate plot
# Note: sort_idx_weak and sort_idx_strong are ascending (best first)
rejection_rates_weak = []
bad_rates_in_approved_weak = []
sorted_labels_weak = true_labels[sort_idx_weak]

for reject_count in range(0, N, 50):  # Sample every 50 for smoother curve
    approved = sorted_labels_weak[: N - reject_count]
    if len(approved) == 0:
        continue
    rejection_rate = reject_count / N
    bad_rate_approved = approved.sum() / len(approved)
    rejection_rates_weak.append(rejection_rate)
    bad_rates_in_approved_weak.append(bad_rate_approved)

rejection_rates_strong = []
bad_rates_in_approved_strong = []
sorted_labels_strong = true_labels[sort_idx_strong]

for reject_count in range(0, N, 50):
    approved = sorted_labels_strong[: N - reject_count]
    if len(approved) == 0:
        continue
    rejection_rate = reject_count / N
    bad_rate_approved = approved.sum() / len(approved)
    rejection_rates_strong.append(rejection_rate)
    bad_rates_in_approved_strong.append(bad_rate_approved)

approval_rates_weak_curve = [1 - r for r in rejection_rates_weak]
approval_rates_strong_curve = [1 - r for r in rejection_rates_strong]

# Figure 1: CAP curve + Profit vs Approval Rate
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=False, dpi=500)

# LEFT: CAP curve
perfect_x = [0, TRUE_BAD_RATE, 1]
perfect_y = [0, 1, 1]
ax1.plot(perfect_x, perfect_y, color=IDEAL_COLOR, label="Ideal")
ax1.plot(
    [0, 1], [0, 1], linestyle="dotted", color=RANDOM_COLOR, alpha=0.5, label="Random"
)
ax1.plot(
    pop_frac_weak,
    recall_weak,
    label=f"Model A Gini: {gini_weak:.2%}",
    color=COLORS[0],
)
ax1.plot(
    pop_frac_strong,
    recall_strong,
    label=f"Model B Gini: {gini_strong:.2%}",
    color=COLORS[1],
)
ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.set_xlabel("Fraction of population")
ax1.set_ylabel("Fraction of bad credit risk (recall)")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
ax1.set_title("Cumulative Accuracy Profile (CAP)", fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(True, which="both", linestyle="dotted", linewidth=0.7, alpha=0.6)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.05)

# RIGHT: Approval rate (x) vs Total profit (y), optimal points marked
ax2.plot(
    approval_curve_weak,
    profit_curve_weak,
    label="Model A",
    color=COLORS[0],
)
ax2.plot(
    approval_curve_strong,
    profit_curve_strong,
    label="Model B",
    color=COLORS[1],
)
ax2.scatter(
    [approval_opt_weak],
    [profit_opt_weak],
    s=30,
    color=COLORS[0],
    zorder=5,
)
ax2.scatter(
    [approval_opt_strong],
    [profit_opt_strong],
    s=30,
    color=COLORS[1],
    zorder=5,
)
ax2.axvline(
    x=approval_opt_weak, color=COLORS[0], linestyle=":", linewidth=1.5, alpha=0.7
)
ax2.axvline(
    x=approval_opt_strong, color=COLORS[1], linestyle=":", linewidth=1.5, alpha=0.7
)

ax2.set_xticks(np.arange(0, 1.1, 0.1))
ax2.set_xlabel("Approval rate")
ax2.set_ylabel("Total profit")
ax2.set_title("Profit vs Approval Rate", fontsize=12)
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(True, which="both", linestyle="dotted", linewidth=0.7, alpha=0.6)
ax2.set_xlim(0, 1)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))


def _profit_fmt(y: float, _) -> str:
    """Format profit values for axis labels.

    Args:
        y: Value to format.
        _: Position parameter (unused).

    Returns:
        Formatted string (e.g., "3.7M" or "150k").
    """
    return f"{y / 1e6:.1f}M" if abs(y) >= 1e6 else f"{y / 1e3:.0f}k"


ax2.yaxis.set_major_formatter(plt.FuncFormatter(_profit_fmt))

fig.suptitle("Economic Value of Gini Based on Unit Economics", fontsize=18)
plt.tight_layout()
fig.subplots_adjust(bottom=0.18)

# Caption for Figure 1
caption1 = (
    f"Model B: +{extra_profit_pct:.0f}% profit at a similar approval rate "
    f"({approval_opt_weak:.0%} vs {approval_opt_strong:.0%})"
)
fig.text(
    0.5,
    0,
    caption1,
    fontsize=10,
    fontweight="bold",
    ha="center",
    va="bottom",
    transform=fig.transFigure,
)

plt.savefig(
    "ap_curve_profit_optimal.png",
    dpi=150,
    bbox_inches="tight",
)
print("\nFigure 1: Profit Optimization")
print("Visualization saved: ap_curve_profit_optimal.png")
print(
    f"Model A optimal: {approval_opt_weak:.1%} approval, {bad_rate_opt_weak:.1%} bad rate, "
    f"profit {profit_opt_weak / 1e6:.2f}M"
)
print(
    f"Model B optimal: {approval_opt_strong:.1%} approval, {bad_rate_opt_strong:.1%} bad rate, "
    f"profit {profit_opt_strong / 1e6:.2f}M"
)
print(f"Model B: +{extra_approval_pct:.1f}% approval, +{extra_profit_pct:.0f}% profit")

# Figure 2: CAP curve + Approval Rate vs Bad Rate
fig2, (ax1_2, ax2_2) = plt.subplots(
    nrows=1, ncols=2, figsize=(9, 4), sharey=False, dpi=500
)

# LEFT: CAP curve (same as Figure 1)
perfect_x = [0, TRUE_BAD_RATE, 1]
perfect_y = [0, 1, 1]
ax1_2.plot(perfect_x, perfect_y, color=IDEAL_COLOR, label="Ideal")
ax1_2.plot(
    [0, 1], [0, 1], linestyle="dotted", color=RANDOM_COLOR, alpha=0.5, label="Random"
)
ax1_2.plot(
    pop_frac_weak,
    recall_weak,
    label=f"Model A Gini: {gini_weak:.2%}",
    color=COLORS[0],
)
ax1_2.plot(
    pop_frac_strong,
    recall_strong,
    label=f"Model B Gini: {gini_strong:.2%}",
    color=COLORS[1],
)
ax1_2.set_xticks(np.arange(0, 1.1, 0.1))
ax1_2.set_yticks(np.arange(0, 1.1, 0.1))
ax1_2.set_xlabel("Fraction of population")
ax1_2.set_ylabel("Fraction of bad credit risk (recall)")
ax1_2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
ax1_2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
ax1_2.set_title("Cumulative Accuracy Profile (CAP)", fontsize=12)
ax1_2.legend(loc="lower right", fontsize=10)
ax1_2.grid(True, which="both", linestyle="dotted", linewidth=0.7, alpha=0.6)
ax1_2.set_xlim(0, 1)
ax1_2.set_ylim(0, 1.05)

# RIGHT: Approval rate vs Bad rate in approved book
ax2_2.plot(
    approval_rates_weak_curve,
    bad_rates_in_approved_weak,
    label="Model A",
    color=COLORS[0],
)
ax2_2.plot(
    approval_rates_strong_curve,
    bad_rates_in_approved_strong,
    label="Model B",
    color=COLORS[1],
)

# Target bad rate: we choose approval cutoff where curve hits this line
ax2_2.axhline(
    y=TARGET_BAD_RATE,
    color=RED_COLOR,
    linestyle="--",
    linewidth=1.5,
    label=f"Risk Appetite ({TARGET_BAD_RATE:.1%})",
)

# Cutoff: approve up to this rate to get TARGET_BAD_RATE in approved book
ax2_2.axvline(
    x=approval_weak_cutoff, color=COLORS[0], linestyle=":", linewidth=2, alpha=0.7
)
ax2_2.axvline(
    x=approval_strong_cutoff, color=COLORS[1], linestyle=":", linewidth=2, alpha=0.7
)

ax2_2.set_xticks(np.arange(0, 1.1, 0.1))
ax2_2.set_yticks(np.linspace(0, 0.3, 7))
ax2_2.set_xlabel("Approval rate")
ax2_2.set_ylabel("Bad rate in approved population")
ax2_2.set_title("Approval Rate vs Risk Appetite", fontsize=12)
ax2_2.legend(loc="upper right", fontsize=10)
ax2_2.grid(True, which="both", linestyle="dotted", linewidth=0.7, alpha=0.6)
ax2_2.set_xlim(0, 1)
ax2_2.set_ylim(0, 0.3)
ax2_2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
ax2_2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))

fig2.suptitle("Economic Value of Gini Based on Risk Appetite", fontsize=18)
plt.tight_layout()
fig2.subplots_adjust(bottom=0.18)

# Caption for Figure 2
extra_approval_pct_cutoff = (approval_strong_cutoff - approval_weak_cutoff) * 100
extra_volume = (n_approved_strong_cutoff - n_approved_weak_cutoff) * LOAN_SIZE
#
caption2 = (
    f"Model B @ {TARGET_BAD_RATE:.1%} risk appetite: "
    f"Model B: +{extra_approval_pct_cutoff:.1f}% approval, +{extra_volume / 1e6:.2f}M volume"
)

fig2.text(
    0.5,
    0,
    caption2,
    fontsize=10,
    fontweight="bold",
    ha="center",
    va="bottom",
    transform=fig2.transFigure,
)

plt.savefig(
    "ap_curve_cutoff_selection.png",
    dpi=600,
    bbox_inches="tight",
)

print("\nFigure 2: Risk Appetite / Cutoff Selection")
print("Visualization saved: ap_curve_cutoff_selection.png")

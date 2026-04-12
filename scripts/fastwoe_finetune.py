"""
WOE recalibration via fastwoe.finetune()

Generates a dumbbell chart showing base vs finetuned WOE per bin
for a single risk driver (Debt-to-Income), with confidence intervals.
"""
# /// script
# dependencies = [
#   "fastwoe==0.1.7",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pycaleva",
#   "scipy",
# ]
# ///

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycaleva import CalibrationEvaluator

from fastwoe import FastWoe

OUT_DIR = Path(__file__).resolve().parent

LOGIT_SCALE = 0.22
N_SAMPLES = 60_000
RNG_SEED = 42

FEATURES: dict[str, list[tuple[str, float, float]]] = {
    "Debt-to-Income": [
        ("<=15%", 40, 960),
        ("15-25%", 80, 720),
        ("25-35%", 150, 550),
        ("35-50%", 220, 380),
        (">50%", 310, 190),
    ],
    "Months on Books": [
        ("<6m", 200, 300),
        ("6-12m", 160, 440),
        ("12-24m", 120, 580),
        ("24-48m", 80, 620),
        (">48m", 40, 760),
    ],
    "Bureau Score": [
        ("<550", 280, 220),
        ("550-620", 180, 420),
        ("620-680", 100, 600),
        ("680-750", 60, 640),
        (">750", 30, 810),
    ],
    "Prior Defaults": [
        ("0", 80, 920),
        ("1", 160, 540),
        ("2", 240, 360),
        ("3+", 320, 180),
    ],
}

TARGET_FEATURE = "Debt-to-Income"

NEW_PERFORMANCE: dict[str, list[tuple[str, float, float]]] = {
    "Debt-to-Income": [
        ("<=15%", 35, 965),
        ("15-25%", 95, 705),
        ("25-35%", 190, 510),
        ("35-50%", 280, 320),
        (">50%", 370, 130),
    ],
}

#-----------------------------------------------------------------------------------------------
# Plotting palette
#-----------------------------------------------------------------------------------------------
C_BASE = "#1d4ed8"
C_ADAPT = "#ea580c"
C_MUTED = "#64748b"
C_GRID = "#e2e8f0"


def configure_style() -> None:
    """Configure the plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": C_MUTED,
            "axes.labelcolor": "#0f172a",
            "axes.titlecolor": "#0f172a",
            "text.color": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": C_GRID,
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Avenir",
                "DejaVu Sans",
                "Helvetica Neue",
                "Arial",
                "sans-serif",
            ],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": C_GRID,
        }
    )

#-----------------------------------------------------------------------------------------------
# Synthetic data
#-----------------------------------------------------------------------------------------------
def _logit(p: float) -> float:
    """Calculate the logit of a probability."""
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return float(np.log(p / (1.0 - p)))


def global_event_rate(
    feature_defs: dict[str, list[tuple[str, float, float]]],
) -> float:
    """Calculate the global event rate from feature definitions."""
    total_bad = total_good = 0.0
    for bins in feature_defs.values():
        for _, bad, good in bins:
            total_bad += bad
            total_good += good
    return float(total_bad / (total_bad + total_good))


def woe_bin_table(bins: list[tuple[str, float, float]], p_prior: float) -> dict[str, float]:
    """Calculate the WOE for each bin."""
    lp0 = _logit(p_prior)
    return {lab: _logit(bad / (bad + good)) - lp0 for lab, bad, good in bins}


def marginal_labels_probs(
    bins: list[tuple[str, float, float]],
) -> tuple[list[str], np.ndarray]:
    """Calculate the marginal labels and probabilities."""
    labels = [b[0] for b in bins]
    w = np.array([b[1] + b[2] for b in bins], dtype=float)
    return labels, w / w.sum()


def make_synthetic_panel(
    n: int,
    feature_defs: dict[str, list[tuple[str, float, float]]],
    target_feature: str,
    rng: np.random.Generator,
    *,
    target_bins: list[tuple[str, float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic panel of data."""
    p_prior = global_event_rate(feature_defs)
    bins_for_target = target_bins or feature_defs[target_feature]

    woe_by_feature = {f: woe_bin_table(bs, p_prior) for f, bs in feature_defs.items()}
    woe_by_feature[target_feature] = woe_bin_table(bins_for_target, p_prior)

    marg: dict[str, tuple[list[str], np.ndarray]] = {
        f: marginal_labels_probs(bs) for f, bs in feature_defs.items()
    }
    marg[target_feature] = marginal_labels_probs(bins_for_target)

    feature_order = list(feature_defs.keys())
    rows: list[list[Any]] = []
    for _ in range(n):
        levels: dict[str, str] = {}
        score = 0.0
        for f in feature_order:
            labels, pr = marg[f]
            cat = str(rng.choice(labels, p=pr))
            levels[f] = cat
            score += woe_by_feature[f][cat]
        logit_p = _logit(p_prior) + LOGIT_SCALE * score
        py = 1.0 / (1.0 + np.exp(-logit_p))
        y = int(rng.uniform() < py)
        rows.append([levels[f] for f in feature_order] + [y])

    df = pd.DataFrame(rows, columns=feature_order + ["y"])
    X = df[feature_order].astype(str)
    y_s = df["y"].astype(np.int8)
    return X, y_s


#-----------------------------------------------------------------------------------------------
# Plotting
#-----------------------------------------------------------------------------------------------
def plot_woe_recalibration(
    base_fw: FastWoe,
    finetuned_fw: FastWoe,
    feature: str,
    label_order: list[str],
    path: Path,
) -> None:
    """Dumbbell chart: base vs finetuned WOE with confidence intervals."""
    m0 = base_fw.mappings_[feature]
    m1 = finetuned_fw.mappings_[feature]

    labels = label_order
    n_bins = len(labels)
    y = np.arange(n_bins)

    w0 = np.array([float(m0.loc[lab, "woe"]) for lab in labels])
    w1 = np.array([float(m1.loc[lab, "woe"]) for lab in labels])

    ci0_lo = np.array([float(m0.loc[lab, "woe_ci_lower"]) for lab in labels])
    ci0_hi = np.array([float(m0.loc[lab, "woe_ci_upper"]) for lab in labels])
    ci1_lo = np.array([float(m1.loc[lab, "woe_ci_lower"]) for lab in labels])
    ci1_hi = np.array([float(m1.loc[lab, "woe_ci_upper"]) for lab in labels])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    # Connectors
    for i in y:
        ax.plot(
            [w0[i], w1[i]],
            [i, i],
            color=C_GRID,
            linewidth=2.4,
            zorder=1,
            solid_capstyle="round",
        )

    # CI whiskers
    for i in y:
        ax.plot(
            [ci0_lo[i], ci0_hi[i]],
            [i - 0.08, i - 0.08],
            color=C_BASE,
            linewidth=1.2,
            alpha=0.4,
            zorder=2,
            solid_capstyle="round",
        )
        ax.plot(
            [ci1_lo[i], ci1_hi[i]],
            [i + 0.08, i + 0.08],
            color=C_ADAPT,
            linewidth=1.2,
            alpha=0.4,
            zorder=2,
            solid_capstyle="round",
        )

    # Points
    ax.scatter(
        w0,
        y - 0.08,
        s=85,
        color=C_BASE,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
        label="Base WOE",
    )
    ax.scatter(
        w1,
        y + 0.08,
        s=85,
        color=C_ADAPT,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
        label="Finetuned WOE",
    )

    ax.axvline(0.0, color=C_MUTED, linewidth=1.0, linestyle=":", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Weight of Evidence")
    ax.set_title(f"{feature}: WOE recalibration", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="x", alpha=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_score_distribution(
    base_fw: FastWoe,
    finetuned_fw: FastWoe,
    X: pd.DataFrame,
    y: pd.Series,
    path: Path,
) -> None:
    """Plot score distribution for base and finetuned models."""
    # Scorecard points: Offset - Factor * WOE_sum (lower risk = higher score)
    OFFSET = 500
    FACTOR = 100
    score_base = OFFSET - FACTOR * base_fw.transform(X).sum(axis=1)
    score_fine = OFFSET - FACTOR * finetuned_fw.transform(X).sum(axis=1)

    prob_base = base_fw.predict_proba(X)[:, 1]
    prob_fine = finetuned_fw.predict_proba(X)[:, 1]

    all_scores = pd.concat([score_base, score_fine])
    score_min = float(all_scores.min())
    score_max = float(all_scores.max())
    num_bins = 20
    bin_edges = np.linspace(score_min, score_max, num_bins + 1)

    is_bad = y == 1
    y_arr = np.asarray(y, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

    for ax, scores, probs, title, z_color in zip(
        axes,
        [score_base, score_fine],
        [prob_base, prob_fine],
        ["Base scorecard", "Finetuned scorecard"],
        ["#FA5252", "#4AA7CD"],
    ):
        ax.hist(
            scores[~is_bad],
            bins=bin_edges,
            label="Good",
            color="#52a1ec",
            edgecolor="#0b62af",
            alpha=0.5,
        )
        ax.hist(
            scores[is_bad],
            bins=bin_edges,
            label="Bad",
            color="#fe595f",
            edgecolor="#cc212e",
            alpha=0.5,
        )

        ce = CalibrationEvaluator(y_arr, probs, outsample=True)
        z = ce.z_test()
        ax.annotate(
            f"Spiegelhalter Z = {z.statistic:.2f} (p = {z.pvalue:.3f})",
            xy=(0.5, 0.96),
            xycoords="axes fraction",
            ha="center",
            fontsize=9,
            color=z_color,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Credit score")
        ax.legend(fontsize=10, frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    max_y = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_y + 200)
    axes[0].set_ylabel("Count")
    fig.suptitle(
        "Score distribution: base vs finetuned",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    configure_style()
    rng = np.random.default_rng(RNG_SEED)

    # Generate synthetic data
    X_dev, y_dev = make_synthetic_panel(N_SAMPLES, FEATURES, TARGET_FEATURE, rng)
    X_new, y_new = make_synthetic_panel(
        N_SAMPLES,
        FEATURES,
        TARGET_FEATURE,
        rng,
        target_bins=NEW_PERFORMANCE[TARGET_FEATURE],
    )

    # Fit base FastWoe model
    base_fw = FastWoe(random_state=RNG_SEED).fit(X_dev, y_dev)

    # Fine-tune one feature
    finetuned_fw = copy.deepcopy(base_fw)
    finetuned_fw.finetune(X_new[[TARGET_FEATURE]], y_new)

    # Print comparison of base and finetuned WOE
    label_order = [b[0] for b in FEATURES[TARGET_FEATURE]]

    print(f"{TARGET_FEATURE}: base vs finetuned WOE")

    m0 = base_fw.mappings_[TARGET_FEATURE]
    m1 = finetuned_fw.mappings_[TARGET_FEATURE]

    print(f"{'Bin':<10} {'Base WOE':>10} {'New WOE':>10} {'Delta':>10}")
    print("-" * 44)
    for lab in label_order:
        w0 = float(m0.loc[lab, "woe"])
        w1 = float(m1.loc[lab, "woe"])
        print(f"{lab:<10} {w0:>10.4f} {w1:>10.4f} {w1 - w0:>+10.4f}")

    fs0 = base_fw.feature_stats_[TARGET_FEATURE]
    fs1 = finetuned_fw.feature_stats_[TARGET_FEATURE]
    print(f"\nIV:  base = {fs0['iv']:.4f}  ->  finetuned = {fs1['iv']:.4f}")
    print(f"Gini: base = {fs0['gini']:.4f}  ->  finetuned = {fs1['gini']:.4f}")

    #-----------------------------------------------------------------------------------------------
    # Plot WOE recalibration and score distribution
    #-----------------------------------------------------------------------------------------------
    plot_woe_recalibration(
        base_fw,
        finetuned_fw,
        TARGET_FEATURE,
        label_order,
        OUT_DIR / "viz_woe_recalibration.png",
    )
    plot_score_distribution(
        base_fw,
        finetuned_fw,
        X_new,
        y_new,
        OUT_DIR / "viz_score_distribution.png",
    )


if __name__ == "__main__":
    main()

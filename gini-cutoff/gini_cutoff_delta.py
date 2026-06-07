# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
#     "catboost",
# ]
# ///
"""Cut-off effect shown as Gini delta vs the full-population Gini.

Same sweep as gini_cutoff.py, but the line is Gini(cut-off) - Gini(full),
in percentage points, to isolate how much discrimination the cut-off costs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

DATA_URL = (
    "https://raw.githubusercontent.com/georgianastasov/"
    "credit-bureau-2021-experian/main/score-model/BankCaseStudyData.csv"
)

LABEL = "Final_Decision"
NUM_FEATURES = [
    "Application_Score",
    "Bureau_Score",
    "Loan_Amount",
    "Time_with_Bank",
    "Time_in_Employment",
    "Loan_to_income",
    "Gross_Annual_Income",
]
CAT_FEATURES = [
    "Loan_Payment_Frequency",
    "Residential_Status",
    "Cheque_Card_Flag",
    "Existing_Customer_Flag",
    "Home_Telephone_Number",
]
FEATURES = CAT_FEATURES + NUM_FEATURES


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load development/validation splits from the public CSV URL."""
    dataset = pd.read_csv(DATA_URL)
    dataset[LABEL] = dataset[LABEL].map({"Accept": 0, "Decline": 1})

    ix_train = dataset["split"] == "Development"
    ix_test = dataset["split"] == "Validation"

    X_train = dataset.loc[ix_train, FEATURES].copy()
    y_train = dataset.loc[ix_train, LABEL].to_numpy()
    X_test = dataset.loc[ix_test, FEATURES].copy()
    y_test = dataset.loc[ix_test, LABEL].to_numpy()

    X_train[CAT_FEATURES] = X_train[CAT_FEATURES].astype(str).fillna("NA")
    X_test[CAT_FEATURES] = X_test[CAT_FEATURES].astype(str).fillna("NA")

    return X_train, y_train, X_test, y_test


def train_model(X_train: pd.DataFrame, y_train: np.ndarray) -> CatBoostClassifier:
    """Fit a CatBoost classifier with native categorical handling."""
    model = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        cat_features=CAT_FEATURES,
        verbose=False,
        random_seed=42,
    )
    model.fit(X_train, y_train)
    return model


def shap_to_scores(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    pdo: float = 50,
    target_points: float = 600,
    target_odds: float = 19,
) -> np.ndarray:
    """Convert CatBoost SHAP values to credit scores (SAS-style scaling)."""
    pool = Pool(X, cat_features=CAT_FEATURES)
    shap_full = model.get_feature_importance(type="ShapValues", data=pool)
    shap_values = shap_full[:, :-1]
    base_value = float(np.mean(shap_full[:, -1]))

    factor = pdo / np.log(2)
    offset = target_points - factor * np.log(target_odds)
    intercept_scaled = factor * base_value

    n_features = shap_values.shape[1]
    per_feature = ((-intercept_scaled) + offset) / n_features
    feature_scores = factor * (-shap_values) + per_feature

    return feature_scores.sum(axis=1)


def gini_vs_cutoff(
    scores: np.ndarray, labels: np.ndarray, step: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep score cut-offs on a per-`step` grid; return (cutoffs, gini, pct_accepted)."""
    lo = int(np.floor(scores.min() / step) * step)
    hi = int(np.ceil(scores.max() / step) * step)
    thresholds = np.arange(lo, hi, step, dtype=float)

    ginis, pcts = [], []
    for t in thresholds:
        mask = scores >= t
        accepted = labels[mask]
        if mask.sum() >= 20 and len(np.unique(accepted)) == 2:
            # Higher score = lower risk, so rank risk by -score.
            ginis.append(2.0 * roc_auc_score(accepted, -scores[mask]) - 1.0)
        else:
            ginis.append(np.nan)
        pcts.append(mask.mean())
    return thresholds, np.array(ginis), np.array(pcts)


plt.rcParams["font.family"] = "Avenir"
# set fontsize to 14 for title and 11 for axes labels and ticks
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
_SCRIPT_DIR = Path(__file__).resolve().parent
_C_CURVE = "#DFFF00"
_C_CUTOFF = "#ff7ebf"
_FIG_GRID = "#d9d9d9"
_GOOD_GRID = "#40B5AD"
_BAD_GRID = "#fe595f"


def plot_delta(scores: np.ndarray, labels: np.ndarray) -> None:
    """Save gini_cutoff_delta.png: Gini drop vs full population, per cut-off."""
    cutoffs, ginis, pcts = gini_vs_cutoff(scores, labels)
    g_full = 2.0 * roc_auc_score(labels, -scores) - 1.0
    delta = (ginis - g_full) * 100  # percentage points, <= 0

    idx50 = int(np.argmin(np.abs(pcts - 0.50)))
    c50, d50 = float(cutoffs[idx50]), delta[idx50]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax_hist = ax.twinx()
    score_min = scores.min().min()
    score_max = scores.max().max()

    # Define the number of bins and bin width
    num_bins = 50
    bin_width = (score_max - score_min) / num_bins
    ax_hist.hist(
        scores[labels == 0],
        bins=np.arange(score_min, score_max + bin_width, bin_width),
        color=_GOOD_GRID,  # "0.78",
        zorder=0,
        # edgecolor="black",
        alpha=0.5,
        label="Goods",
    )
    ax_hist.hist(
        scores[labels == 1],
        bins=np.arange(score_min, score_max + bin_width, bin_width),
        color=_BAD_GRID,  # "0.35",
        zorder=1,
        alpha=0.5,
        # edgecolor="black",
        label="Bads",
    )
    ax_hist.set_ylabel("Applicants per score bin", color="0.5")
    ax_hist.tick_params(axis="y", colors="0.5")
    ax_hist.legend(
        bbox_to_anchor=(0.02, 0.95),
        loc="upper left",
        framealpha=0.9,
        edgecolor="none",
    )
    ax_hist.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    ax.axhline(0, color="0.4", linewidth=1.0, zorder=2)
    ax.plot(
        cutoffs,
        delta,
        color=_C_CURVE,
        linewidth=2.6,
        marker=None,
        markersize=5,
        zorder=3,
        label="Gini drop vs full population",
    )
    ax.legend(
        bbox_to_anchor=(0.5, 0.9),
        loc="center",
        framealpha=0.9,
        edgecolor="none",
    )
    ax.scatter([c50], [d50], s=55, color=_C_CURVE, zorder=5, edgecolors="none")
    ax.axvline(c50, color=_C_CUTOFF, linewidth=2.0, zorder=2)

    ax.annotate(
        f"Cut-off {c50:.0f}:\n{d50:.1f} pts of Gini lost",
        xy=(c50, d50),
        xytext=(c50 - 8, d50 + 6),
        fontweight="bold",
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        ha="right",
    )

    ax.set_xlabel("Credit score cut-off (accept if score ≥ cut-off)")
    ax.set_ylabel("Gini vs full population Gini (pts)")
    ax.set_title(
        "How much Gini do the cut-off cost?",
        pad=10,
        color="black",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}"))

    ax.grid(True, linestyle="-", linewidth=0.75, color=_FIG_GRID, alpha=0.95, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#7f7f7f")
        spine.set_linewidth(0.9)

    out = _SCRIPT_DIR / "gini_cutoff_delta.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Full-population Gini: {g_full:.1%}")
    print(f"Cut-off {c50:.0f}: {d50:.1f} pts lost")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = train_model(X_train, y_train)
    scores = shap_to_scores(model, X_test)
    plot_delta(scores, y_test)

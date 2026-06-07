# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
#     "catboost",
#     "fastwoe",
# ]
# ///
"""Visual mechanism behind the cut-off Gini drop.

Two panels, both over the same per-100 cut-off grid:
  Left  - who survives: bad rate of the accepted book and the share of all
          bads still retained (bads vanish faster than goods).
  Right - why Gini falls: Somers' D against the discordant-pair share, which
          rises because only hard, high-scoring bads survive.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from fastwoe.metrics import somersd_yx
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


def cutoff_pair_table(
    scores: np.ndarray, labels: np.ndarray, step: int = 100
) -> pd.DataFrame:
    """Build a row-per-cut-off table of Somers' D and concordant/discordant pair counts."""
    cutoffs, _, _ = gini_vs_cutoff(scores, labels, step=step)

    rows = []
    for t in cutoffs:
        mask = scores >= t
        y = labels[mask]
        n, bads = int(mask.sum()), int(y.sum())
        goods = n - bads

        if n >= 20 and len(np.unique(y)) == 2:
            r = somersd_yx(y, -scores[mask])
            rows.append(
                {
                    "cutoff": int(t),
                    "n": n,
                    "goods": goods,
                    "bads": bads,
                    "pct_accept": mask.mean(),
                    "somers_d": r.statistic,
                    "concordant": r.concordant_pairs,
                    "discordant": r.discordant_pairs,
                    "ties": r.ties,
                    "good_bad_pairs": goods * bads,
                }
            )

    df = pd.DataFrame(rows)
    df["conc_minus_disc"] = df["concordant"] - df["discordant"]
    df["disc_share"] = df["discordant"] / (df["concordant"] + df["discordant"])
    return df


plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams.update(
    {
        "font.family": "Avenir",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#7f7f7f",
        "axes.linewidth": 0.9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_C_BLUE = "#1e90ff"
_C_RED = "#fe595f"
_C_TEAL = "#E4D00A"
_C_CUTOFF = "#ff7ebf"
_FIG_GRID = "#d9d9d9"


def _style(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="-", linewidth=0.75, color=_FIG_GRID, alpha=0.95, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#7f7f7f")
        spine.set_linewidth(0.9)


def plot_mechanism(scores: np.ndarray, labels: np.ndarray) -> None:
    """Save gini_cutoff_mechanism.png."""
    df = cutoff_pair_table(scores, labels)
    cutoffs = df["cutoff"].to_numpy()
    total_bads = int(labels.sum())

    bad_rate = (df["bads"] / df["n"] * 100).to_numpy()
    bads_retained = (df["bads"] / total_bads * 100).to_numpy()
    somers = (df["somers_d"] * 100).to_numpy()
    disc_share = (
        df["discordant"] / (df["concordant"] + df["discordant"]) * 100
    ).to_numpy()

    c50 = float(cutoffs[int(np.argmin(np.abs(df["pct_accept"].to_numpy() - 0.50)))])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.subplots_adjust(wspace=0.32)

    # --- Left: who survives the cut-off ---
    axL.plot(
        cutoffs,
        bads_retained,
        color=_C_RED,
        lw=2.6,
        marker="o",
        ms=5,
        label="Share of all bads retained",
    )
    axL.plot(
        cutoffs,
        bad_rate,
        color=_C_TEAL,
        lw=2.6,
        marker="s",
        ms=5,
        label="Bad rate of accepts",
    )
    axL.axvline(c50, color=_C_CUTOFF, lw=2.0, zorder=2)
    axL.set_xlabel("Credit score cut-off (accept if score ≥ cut-off)")
    axL.set_ylabel("Percent")
    axL.set_title("Who survives the cut-off", pad=10)
    axL.yaxis.set_major_formatter(mticker.PercentFormatter())
    axL.set_ylim(0, 105)
    axL.legend(
        framealpha=0.9, edgecolor="none", bbox_to_anchor=(0.95, 1), loc="upper right"
    )
    _style(axL)

    # --- Right: why Gini falls ---
    axR.plot(cutoffs, somers, color=_C_BLUE, lw=2.6, marker="o", ms=5, label="Gini")
    axR.plot(
        cutoffs,
        disc_share,
        color=_C_RED,
        lw=2.6,
        marker="s",
        ms=5,
        label="Discordant pair share",
    )
    axR.axvline(c50, color=_C_CUTOFF, lw=2.0, zorder=2)
    axR.set_xlabel("Credit score cut-off (accept if score ≥ cut-off)")
    axR.set_ylabel("Percent")
    axR.set_title("Why Gini falls: discordant pairs take over", pad=10)
    axR.yaxis.set_major_formatter(mticker.PercentFormatter())
    axR.set_ylim(0, 100)
    axR.legend(
        framealpha=0.9, edgecolor="none", bbox_to_anchor=(0.95, 1), loc="upper right"
    )
    _style(axR)

    out = _SCRIPT_DIR / "gini_cutoff_mechanism.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out}")
    print(df[["cutoff", "bads", "somers_d", "discordant"]].to_string(index=False))


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = train_model(X_train, y_train)
    scores = shap_to_scores(model, X_test)
    plot_mechanism(scores, y_test)

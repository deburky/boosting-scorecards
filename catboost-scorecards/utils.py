"""utils.py."""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from catboost_scorecard import CatBoostScorecard
from sklearn.base import BaseEstimator, TransformerMixin


# pylint: disable=invalid-name
class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocess high-cardinality categorical features for interpretable CatBoost models.
    You can control either:
      - max_categories (top N by frequency)
      - top_p (top N% cumulative frequency)
    """

    def __init__(self, max_categories=None, top_p=0.9, other_token="__other__"):
        assert max_categories or top_p, "Set either `max_categories` or `top_p`"
        self.max_categories = max_categories
        self.top_p = top_p
        self.other_token = other_token
        self.category_maps = {}
        self.cat_features_ = None

    def fit(self, X: pd.DataFrame, y=None, cat_features: list[str] = None):
        """Fit the preprocessor to the DataFrame."""
        self.cat_features_ = (
            cat_features or X.select_dtypes(include="object").columns.tolist()
        )

        for col in self.cat_features_:
            vc = (
                X[col]
                .astype(str)
                .value_counts(dropna=False)
                .sort_values(ascending=False)
            )
            if self.top_p is not None:
                cumulative = vc.cumsum() / vc.sum()
                top_cats = cumulative[cumulative <= self.top_p].index.tolist()
            else:
                top_cats = vc.nlargest(self.max_categories).index.tolist()

            self.category_maps[col] = set(top_cats)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame by replacing low-frequency categories."""
        X_ = X.copy()
        for col in self.cat_features_:
            allowed = self.category_maps[col]
            X_[col] = (
                X_[col]
                .astype(str)
                .apply(
                    lambda x, allowed=allowed: x if x in allowed else self.other_token
                )
            )
        return X_

    # pylint: disable=arguments-differ
    def fit_transform(
        self, X: pd.DataFrame, y=None, cat_features: list[str] = None
    ) -> pd.DataFrame:
        """Fit the preprocessor and transform the data."""
        return self.fit(X, y, cat_features).transform(X)

    def __call__(self, X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
        """A callable interface for the preprocessor."""
        return self.fit_transform(X, cat_features=cat_features)

    def get_mapping(self) -> dict:
        """Get the mapping of categorical features to their top categories."""
        return self.category_maps


# pylint: disable=protected-access
class CatBoostTreeVisualizer:
    """Class to visualize CatBoost trees, handling both feature-name and index-based trees."""

    def __init__(self, scorecard: pd.DataFrame, plot_config: Dict[str, Any] = None):
        self.scorecard = scorecard
        self.tree_dump = {}
        self.plot_config = plot_config or {}

        self.config = {
            "facecolor": "#ffffff",
            "edgecolor": "black",
            "edgewidth": 0,
            "font_size": 12,
            "figsize": (10, 5),
            "level_distance": 2,
            "sibling_distance": 20.0,
        }
        self.config |= self.plot_config

    def build_tree(self, tree_idx: int) -> Dict[str, Any]:  # pylint: disable=too-many-statements, too-many-locals
        """Build a tree structure from the scorecard DataFrame, handling both tree types."""
        tree_df = self.scorecard[self.scorecard["Tree"] == tree_idx].reset_index(
            drop=True
        )
        leaf_count = len(tree_df)
        depth = leaf_count.bit_length() - 1

        # Get the level conditions directly
        level_conditions = None
        feature_names = []

        # Try to get level conditions from CatBoostScorecard
        if (
            hasattr(CatBoostScorecard, "level_conditions")
            and tree_idx in CatBoostScorecard.level_conditions
        ):
            level_conditions = CatBoostScorecard.level_conditions[tree_idx]

        # Check if we have feature names
        if (
            hasattr(CatBoostScorecard, "debug_info")
            and "feature_names" in CatBoostScorecard.debug_info
        ):
            feature_names = CatBoostScorecard.debug_info["feature_names"]

        # Check if this is an index-based tree
        is_index_based = False
        if (
            hasattr(CatBoostScorecard, "debug_info")
            and f"tree_{tree_idx}_is_index_based" in CatBoostScorecard.debug_info
        ):
            is_index_based = CatBoostScorecard.debug_info[
                f"tree_{tree_idx}_is_index_based"
            ]

        # If no conditions available, try to extract from model
        if (
            not level_conditions
            and hasattr(CatBoostScorecard, "model")
            and hasattr(CatBoostScorecard, "pool")
        ):
            cb_obj = CatBoostScorecard.model._object
            level_conditions = list(
                reversed(cb_obj._get_tree_splits(tree_idx, CatBoostScorecard.pool))
            )

            # Check if index-based
            is_index_based = all(
                CatBoostScorecard._is_numeric_only_condition(cond)
                for cond in level_conditions
                if cond
            )

        # If still no conditions, extract them from leaf conditions
        if not level_conditions:
            # Extract conditions from each leaf's condition string
            all_leaf_conditions = []
            for _, row in tree_df.iterrows():
                conditions = row["Conditions"].split(" AND ")
                all_leaf_conditions.append(conditions)

            # Organize by level
            level_conditions = []
            for level in range(depth):
                level_set = set()
                for leaf_conds in all_leaf_conditions:
                    if level < len(leaf_conds):
                        # Extract the condition at this level
                        cond = leaf_conds[level]
                        level_set.add(cond)
                if level_set:
                    # Get one representative condition
                    level_conditions.append(list(level_set)[0])

        def build_node(path: str, level: int) -> Dict[str, Any]:
            if level == depth:
                row = tree_df.iloc[int(path, 2)]
                return {
                    "name": (
                        f"val={row['LeafValue']:.3f}\n"
                        f"woe={row['WOE']:.3f}\n"
                        f"rate={row['EventRate']:.3f}\n"
                        f"count={int(row['Count'])}"
                    ),
                    "depth": level,
                }

            # For non-leaf nodes, get the condition at this level
            display_condition = f"Level {level} split"
            if level_conditions and level < len(level_conditions):
                condition = level_conditions[level]

                # Handle index-based condition
                if (
                    is_index_based
                    and hasattr(CatBoostScorecard, "_is_numeric_only_condition")
                    and CatBoostScorecard._is_numeric_only_condition(condition)
                ):
                    feature_idx = int(condition.strip())
                    if feature_names and 0 <= feature_idx < len(feature_names):
                        # Show feature name if available
                        display_condition = (
                            f"{feature_names[feature_idx]}, value>threshold"
                        )
                    else:
                        # Otherwise just show the index with proper formatting
                        display_condition = f"{feature_idx}, value>threshold"

                # Handle feature-name-based condition
                else:
                    if hasattr(CatBoostScorecard, "_get_split_feature_value"):
                        feature, value, split_type = (
                            CatBoostScorecard._get_split_feature_value(condition)
                        )

                        if split_type == "numerical":
                            # For numerical values, use "feature, value>threshold" format
                            display_condition = f"{feature}, value>{value}"
                        elif split_type == "categorical_value":
                            # Use "feature = 'value'" format instead of "feature, value=value"
                            display_condition = f"{feature}='{value}'"
                        elif split_type == "categorical_list":
                            if isinstance(value, list) and len(value) == 1:
                                display_condition = f"{feature}='{value[0]}'"
                            else:
                                # For multi-value lists, keep current format or adjust as needed
                                cat_str = ", ".join(f"'{c}'" for c in value)
                                display_condition = f"{feature} IN ({cat_str})"

            # Important: Swap the Yes/No branch direction to match CatBoost native visualization
            yes_branch = build_node(
                f"{path}0", level + 1
            )  # 0 = Yes branch (right in native)
            no_branch = build_node(
                f"{path}1", level + 1
            )  # 1 = No branch (left in native)

            return {
                "name": display_condition,
                "depth": level,
                "children": {
                    "Yes": yes_branch,  # Display "Yes" for the 0-branch (right)
                    "No": no_branch,  # Display "No" for the 1-branch (left)
                },
            }

        self.tree_dump[str(tree_idx)] = build_node("", 0)
        return self.tree_dump[str(tree_idx)]

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def _draw_tree(
        self,
        node: Dict[str, Any],
        depth: int = 0,
        pos_x: float = 0.0,
        level_distance: float = None,
        sibling_distance: float = None,
    ) -> None:
        if level_distance is None:
            level_distance = self.config["level_distance"]
        if sibling_distance is None:
            sibling_distance = self.config["sibling_distance"]

        node_pos = (pos_x, -depth * level_distance)
        plt.text(
            node_pos[0],
            node_pos[1],
            node["name"],
            ha="center",
            va="center",
            fontsize=self.config["font_size"],
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc=self.config["facecolor"],
                ec=self.config["edgecolor"],
                linewidth=self.config["edgewidth"],
            ),
        )

        if "children" in node:
            for i, (label, child) in enumerate(node["children"].items()):
                # Important: Swap left/right direction to match CatBoost's convention
                # Yes = right branch, No = left branch
                offset = (1.0 if label == "Yes" else -1.0) * sibling_distance
                child_x = pos_x + offset
                child_y = -((depth + 1) * level_distance)
                plt.plot(
                    [pos_x, child_x],
                    [node_pos[1], child_y],
                    color=self.config["edgecolor"],
                )
                label_x = (pos_x + child_x) / 2
                label_y = (node_pos[1] + child_y) / 2

                # Shift Yes and No slightly
                if label == "Yes":
                    label_x += 1.0  # Move "Yes" slightly to the right
                else:
                    label_x -= 1.0  # Move "No" slightly to the left
                label_y += 0.2

                plt.text(
                    label_x,
                    label_y,
                    label,
                    fontsize=self.config["font_size"] - 2,
                    fontfamily="monospace",
                    ha="center",
                    va="center",
                )

                self._draw_tree(
                    child, depth + 1, child_x, level_distance, sibling_distance / 1.5
                )

    def plot_tree(self, tree_idx: int = 0) -> None:
        """Plot a specific CatBoost tree with dynamic sizing."""
        if str(tree_idx) not in self.tree_dump:
            tree = self.build_tree(tree_idx)
        else:
            tree = self.tree_dump[str(tree_idx)]

        # Estimate actual tree depth (recursive)
        def _max_depth(node):
            if "children" not in node:
                return node.get("depth", 0)
            return max(_max_depth(child) for child in node["children"].values())

        depth = _max_depth(tree)

        # Dynamically scale figsize and spacing
        base_width = 10
        base_height = 5
        width_per_level = 2.5
        sibling_scale = 4.0
        level_scale = 1.2
        horizontal_width = base_width + width_per_level * depth * 2.0

        self.config["figsize"] = (
            horizontal_width,
            base_height,
        )
        self.config["sibling_distance"] = sibling_scale * (depth + 1)
        self.config["level_distance"] = level_scale * (depth + 1)

        # Plot the tree
        plt.figure(figsize=self.config["figsize"], dpi=110)
        plt.axis("off")
        self._draw_tree(
            tree,
            level_distance=self.config["level_distance"],
            sibling_distance=self.config["sibling_distance"],
        )
        plt.title(f"CatBoost Tree {tree_idx + 1}", fontsize=12, fontfamily="monospace")
        plt.show()

    def print_debug_info(self) -> None:
        """Print debug information if available."""
        if hasattr(
            CatBoostScorecard, "debug_info"
        ) and CatBoostScorecard.debug_info.get("debug_enabled", False):
            print("CatBoost Tree Debug Information:")
            for key, value in CatBoostScorecard.debug_info.items():
                if key != "debug_enabled":
                    print(f"{key}: {value}")

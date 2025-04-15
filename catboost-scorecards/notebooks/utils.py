"""utils.py."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
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
    """Class to visualize CatBoost trees with correct branch ordering and accurate split conditions."""

    def __init__(self, scorecard: pd.DataFrame, plot_config: Dict[str, Any] = None):
        self.scorecard = scorecard
        self.tree_cache = {}
        self.plot_config = plot_config or {}

        # Default configuration
        self.config = {
            "facecolor": "#ffffff",
            "edgecolor": "black",
            "edgewidth": 0,
            "font_size": 14,
            "figsize": (18, 10),
            "level_distance": 10.0,
            "sibling_distance": 10.0,
            "fontfamily": "monospace",
            "yes_color": "#1f77b4",  # Blue for "Yes" branches
            "no_color": "#ff7f0e",  # Orange for "No" branches
            "leaf_color": "#2ca02c",  # Green for leaf nodes
        }
        # Update the config with user-defined plot_config
        try:
            self.config |= self.plot_config
        except TypeError:  # This will happen in Python < 3.9
            self.config.update(self.plot_config)

    def _parse_condition(self, condition: str) -> str:
        """Format conditions to accurately represent CatBoost's splitting logic."""
        if " <= " in condition:
            parts = condition.split(" <= ")
            return (
                f"{parts[0]} > {parts[1]}"  # Convert to CatBoost's actual split logic
            )
        elif " > " in condition:
            parts = condition.split(" > ")
            return f"{parts[0]} ≤ {parts[1]}"  # Convert to complement
        elif " in [" in condition:
            cats = condition.split(" in [")[1].split("]")[0]
            return f"{condition.split(' in [')[0]} ∈ {{{cats}}}"
        elif " not in [" in condition:
            cats = condition.split(" not in [")[1].split("]")[0]
            return f"{condition.split(' not in [')[0]} ∉ {{{cats}}}"
        return condition

    def build_tree(self, tree_idx: int) -> dict:
        """Build tree structure with correct CatBoost branch ordering."""
        tree_df = self.scorecard[self.scorecard["Tree"] == tree_idx]
        leaf_count = len(tree_df)
        depth = leaf_count.bit_length() - 1

        path_to_leaf = {format(i, f"0{depth}b"): i for i in range(leaf_count)}

        def build_node(path: str, level: int) -> dict:
            if level == depth:
                leaf_idx = path_to_leaf[path]
                row = tree_df.iloc[leaf_idx]

                return {
                    "name": (
                        f"count: {int(row['Count'])}\n"
                        f"rate: {row['EventRate']:.3f}\n"
                        f"woe: {row['WOE']:.3f}\n"
                        f"val: {row['LeafValue']:.3f}"
                    ),
                    "depth": level,
                    "is_leaf": True,
                }

            sample_leaf = next(k for k in path_to_leaf if k.startswith(path))
            full_condition = tree_df.iloc[path_to_leaf[sample_leaf]]["Conditions"]
            level_condition = full_condition.split(" AND ")[level]

            return {
                "name": self._parse_condition(level_condition),
                "depth": level,
                "children": {
                    # Note: In CatBoost, "Yes" path is when condition is TRUE (feature > threshold)
                    "Yes": build_node(f"{path}1", level + 1),
                    "No": build_node(f"{path}0", level + 1),
                },
            }

        tree_structure = build_node("", 0)
        self.tree_cache[tree_idx] = tree_structure
        return tree_structure

    def _draw_tree(
        self,
        node: Dict[str, Any],
        depth: int = 0,
        pos_x: float = 0.0,
        level_distance: float = None,
        sibling_distance: float = None,
    ) -> None:
        """Draw tree with accurate CatBoost split logic."""
        if level_distance is None:
            level_distance = self.config["level_distance"]
        if sibling_distance is None:
            sibling_distance = self.config["sibling_distance"]

        node_pos = (pos_x, -depth * level_distance)

        # Calculate optimal vertical spacing
        line_height = 0.15 * level_distance
        initial_offset = 0.5 * line_height * (node["name"].count("\n") - 1)

        # Draw each line of text
        for i, line in enumerate(node["name"].split("\n")):
            plt.text(
                node_pos[0],
                node_pos[1] - initial_offset + i * line_height,
                line,
                ha="center",
                va="center",
                fontsize=self.config["font_size"],
                fontfamily=self.config["fontfamily"],
                # add white background
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=self.config["facecolor"],
                    edgecolor=self.config["edgecolor"],
                    linewidth=self.config["edgewidth"],
                ),
            )

        if "children" in node:
            for label, child in node["children"].items():
                offset = (1.0 if label == "Yes" else -1.0) * sibling_distance
                child_x = pos_x + offset
                child_y = -((depth + 1) * level_distance)

                # Draw connection line with appropriate color
                line_color = (
                    self.config["yes_color"]
                    if label == "Yes"
                    else self.config["no_color"]
                )
                plt.plot(
                    [pos_x, child_x],
                    [node_pos[1], child_y],
                    color=line_color,
                    linewidth=1.5,
                    linestyle="-",
                    alpha=0.7,
                )

                # Position branch labels
                label_x = (pos_x + child_x) / 2
                label_y = (node_pos[1] + child_y) / 2

                plt.text(
                    label_x,
                    label_y,
                    label,
                    fontsize=self.config["font_size"] - 1,
                    fontfamily=self.config["fontfamily"],
                    ha="center",
                    va="center",
                    color=line_color,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=self.config["facecolor"],
                        edgecolor=self.config["edgecolor"],
                        linewidth=self.config["edgewidth"],
                    ),
                )

                self._draw_tree(
                    child,
                    depth + 1,
                    child_x,
                    level_distance,
                    sibling_distance / 1.8,
                )

    def plot_tree(self, tree_idx: int = 0, title: str = None) -> None:
        """Plot tree with accurate CatBoost split logic."""
        if tree_idx not in self.tree_cache:
            self.build_tree(tree_idx)

        tree = self.tree_cache[tree_idx]

        # Calculate tree depth
        def get_max_depth(node):
            if "children" not in node:
                return node["depth"]
            return max(get_max_depth(child) for child in node["children"].values())

        depth = get_max_depth(tree)

        # Dynamic sizing
        char_width = 0.018
        max_line_len = max(
            len(line)
            for node in self._flatten_tree(tree)
            for line in node["name"].split("\n")
        )

        fig_width = max(26, max_line_len * char_width * (depth + 2))
        fig_height = 4 + depth * 0.8  # Slightly more compact

        plt.figure(figsize=(fig_width, fig_height), dpi=200)
        plt.axis("off")

        # Calculate optimal level distance based on line count
        max_lines = max(
            node["name"].count("\n") + 1 for node in self._flatten_tree(tree)
        )
        level_distance = 1.2 + 0.15 * max_lines  # More compact spacing

        self._draw_tree(
            tree, level_distance=level_distance, sibling_distance=3.0 * (depth + 1)
        )

        plt.title(
            title or f"CatBoost Tree {tree_idx}",
            fontsize=self.config["font_size"] + 4,
            fontfamily=self.config["fontfamily"],
            pad=20,
        )
        plt.tight_layout()
        plt.show()

    def _flatten_tree(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Helper to flatten tree structure for calculations."""
        nodes = [node]
        if "children" in node:
            for child in node["children"].values():
                nodes.extend(self._flatten_tree(child))
        return nodes

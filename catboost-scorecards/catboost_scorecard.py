# /// script
# dependencies = [
#   "catboost>=1.2",
#   "pandas>=2.0",
#   "pydantic>=2.0",
#   "numpy>=1.0",
#   "scikit-learn>=1.0",
# ]
# ///

"""
CatBoost Scorecard Extraction
=================================
This module provides functionality to extract data for scorecards from CatBoost models.
It handles both numerical and categorical features, including text features.

Author: Denis Burakov
Github: @deburky
License: MIT
This code is licensed under the MIT License.
Copyright (c) 2025 Denis Burakov

"""
from typing import Dict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pydantic import BaseModel, field_validator


# pylint: disable=no-self-argument, too-few-public-methods
class CatBoostScorecard:
    """
    A class for extracting human-readable scorecards from CatBoost models.
    Handles both numerical and categorical features, including text features.
    """

    class _CBScorecardInput(BaseModel):
        """Internal input validation model"""

        model: CatBoostClassifier
        pool: Pool

        class Config:
            """Configuration for Pydantic model"""

            arbitrary_types_allowed = True

        @field_validator("model")
        def validate_model(cls, v):
            """Validate that the model is a CatBoostClassifier."""
            if not isinstance(v, CatBoostClassifier):
                raise TypeError("model must be an instance of CatBoostClassifier.")
            return v

        @field_validator("pool")
        def validate_pool(cls, v):
            """Validate that the pool is a CatBoost Pool object."""
            if not isinstance(v, Pool):
                raise TypeError("pool must be a catboost.Pool object.")
            return v

    @staticmethod
    def _get_leaf_conditions(cb_obj: object, pool: Pool, tree_idx: int) -> Dict[int, str]:
        """Get leaf conditions for a given tree index."""
        split_conditions = cb_obj._get_tree_splits(tree_idx, pool)
        leaf_count = int(cb_obj._get_tree_leaf_counts()[tree_idx])
        tree_depth = leaf_count.bit_length() - 1

        if 2**tree_depth != leaf_count:
            raise ValueError(f"Tree {tree_idx} is not complete binary")

        return {
            leaf_idx: " AND ".join(
                CatBoostScorecard._parse_condition(cond, bit)
                for cond, bit in zip(split_conditions, format(leaf_idx, f"0{tree_depth}b"))
            )
            for leaf_idx in range(leaf_count)
        }

    @staticmethod
    def _parse_condition(condition: str, bit: str) -> str:
        """Parse a condition string into a human-readable format."""
        # Handle numerical splits
        if "bin=" in condition:
            parts = condition.split(", bin=")
            feature = parts[0].strip()
            value = parts[1].strip()
            return f"{feature} {'<=' if bit == '0' else '>'} {value}"

        # Handle categorical/text splits
        elif "[" in condition:
            parts = condition.split("[", 1)
            feature = parts[0].strip()
            categories = parts[1].split("]", 1)[0]
            return f"{feature} {'in' if bit == '0' else 'not in'} [{categories}]"

        # Fallback for other cases
        else:
            feature = condition.split(":", 1)[0].strip() if ":" in condition else condition.strip()
            return f"{feature} {'<=' if bit == '0' else '>'}"

    # pylint: disable=protected-access
    @classmethod
    def extract(cls, model: CatBoostClassifier, pool: Pool) -> pd.DataFrame:
        """
        Extract a scorecard from a trained CatBoost model.

        Args:
            model: Trained CatBoostClassifier
            pool: CatBoost Pool object used for training/validation

        Returns:
            pd.DataFrame: Scorecard with leaf conditions, statistics, and predictions
        """
        # Validate input
        input_data = cls._CBScorecardInput(model=model, pool=pool)
        cb_obj = input_data.model._object
        tree_count = cb_obj._get_tree_count()
        y = input_data.pool.get_label()
        avg_event_rate = float(y.mean())

        # Extract leaf values and conditions
        leaf_records = []
        for tree_idx in range(tree_count):
            leaf_vals = cb_obj._get_tree_leaf_values(tree_idx)
            leaf_conditions = cls._get_leaf_conditions(cb_obj, input_data.pool, tree_idx)
            leaf_count = int(cb_obj._get_tree_leaf_counts()[tree_idx])

            for leaf_idx in range(leaf_count):
                val_str = leaf_vals[leaf_idx] if leaf_idx < len(leaf_vals) else "val = 0.0"
                clean_val = float(val_str.replace("val = ", "").strip())
                leaf_records.append(
                    {
                        "Tree": tree_idx,
                        "LeafIndex": leaf_idx,
                        "LeafValue": clean_val,
                        "Conditions": leaf_conditions.get(leaf_idx, None),
                    }
                )

        leaf_df = pd.DataFrame(leaf_records)

        # Get leaf assignments and aggregate statistics
        leaf_assignments = input_data.model.calc_leaf_indexes(input_data.pool)
        assignments_df = pd.DataFrame(leaf_assignments, columns=[f"tree_{i}" for i in range(leaf_assignments.shape[1])])

        # Aggregate target stats per leaf
        agg_dfs = []
        for tree_idx in range(tree_count):
            temp_df = pd.DataFrame({"Tree": tree_idx, "LeafIndex": assignments_df[f"tree_{tree_idx}"], "Label": y})
            agg = temp_df.groupby(["Tree", "LeafIndex"])["Label"].agg(Events="sum", Count="count").reset_index()
            agg["NonEvents"] = agg["Count"] - agg["Events"]
            agg_dfs.append(agg)

        # Create complete scorecard
        scorecard_df = (
            pd.DataFrame(
                [
                    {"Tree": tree_idx, "LeafIndex": leaf_idx}
                    for tree_idx in range(tree_count)
                    for leaf_idx in range(int(cb_obj._get_tree_leaf_counts()[tree_idx]))
                ]
            )
            .merge(pd.concat(agg_dfs), on=["Tree", "LeafIndex"], how="left")
            .fillna({"Events": 0, "Count": 0, "NonEvents": 0})
            .merge(leaf_df, on=["Tree", "LeafIndex"], how="left")
        )

        # Calculate event rate
        scorecard_df["EventRate"] = (
            scorecard_df["Events"].astype(float) / scorecard_df["Count"].replace(0, np.nan)
        ).fillna(avg_event_rate)

        return scorecard_df.sort_values(["Tree", "LeafIndex"]).reset_index(drop=True)
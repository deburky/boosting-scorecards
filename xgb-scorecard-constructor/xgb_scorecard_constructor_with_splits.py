import json
import numpy as np
import pandas as pd
import xgboost as xgb
import contextlib
from typing import List


class XGBScorecardConstructor:
    """
    Author: Denis Burakov (GitHub: http://github.com/deburky)

    Description:
    A class for generating a scorecard from a trained XGBoost model. The methodology is inspired by the NVIDIA GTC Talk
    "Machine Learning in Retail Credit Risk" by Paul Edwards (GitHub: https://github.com/pedwardsada).

    Parameters:
    - xgb_model (xgboost.XGBClassifier): The trained XGBoost model.
    - X_train (pd.DataFrame): Features of the training data.
    - y_train (pd.Series): Labels of the training data.

    Methods:
    - extract_leaf_weights: Extracts leaf weights based on the XGBoost model.
    - construct_scorecard: Generates the scorecard by combining leaf weights with binning summary.

    Example usage:
    ```python
    # Instantiate the XGBScorecardConstructor
    scorecard_constructor = XGBScorecardConstructor(xgb_model, X_train, y_train)

    # Generate the scorecard
    xgb_scorecard = scorecard_constructor.construct_scorecard()

    # Print the scorecard
    print(xgb_scorecard)
    ```

    The constructed scorecard includes information such as Tree, Node, Feature, Split, Sign, Events,
    NonEvents, Total, EventRate, XAddEvidence, and WOE for each tree and node.

    #TODO: add support for handling leafs with missing values

    """

    def __init__(self, xgb_model, X_train, y_train):
        self.xgb_model = xgb_model
        self.X_train = X_train
        self.y_train = y_train
        self.booster_ = xgb_model.get_booster()
        self.print_splits = False

    def extract_leaf_weights(self):
        tree_df = self.booster_.trees_to_dataframe()

        # Extract relevant columns for feature gains
        feature_gains = tree_df[tree_df["Feature"] != "Leaf"][
            ["Tree", "Node", "ID", "Split", "Yes", "No", "Feature", "Gain"]
        ]

        # Extract relevant columns for leaf gains
        leaf_gains = tree_df[tree_df["Feature"] == "Leaf"][
            ["Tree", "Node", "ID", "Split", "Yes", "No", "Feature", "Gain"]
        ]

        # Helper function for merging and renaming
        def merge_and_rename(gains_df, condition_column, sign):
            condition_df = feature_gains.merge(
                gains_df, left_on=condition_column, right_on="ID", how="inner"
            )
            condition_df = condition_df.rename(
                columns={
                    "Tree_x": "Tree",
                    "Node_y": "Node",
                    "Split_x": "Split",
                    "Feature_x": "Feature",
                    "Gain_y": "XAddEvidence",
                }
            )
            condition_df["Sign"] = sign
            return condition_df[
                ["Tree", "Node", "Feature", "Sign", "Split", "XAddEvidence"]
            ]

        # Merge on 'Yes' and 'No' ID (True = <, False = >=)
        yes_condition_df = merge_and_rename(leaf_gains, "Yes", "<")
        no_condition_df = merge_and_rename(leaf_gains, "No", ">=")

        # Concatenate the DataFrames
        leaf_weights_df = pd.concat(
            [yes_condition_df, no_condition_df], ignore_index=True
        )
        leaf_weights_df = leaf_weights_df.sort_values(by="Tree").reset_index(drop=True)

        return leaf_weights_df

    def construct_scorecard(self, print_splits=False):
        self.print_splits = print_splits
        # # retrieve base score
        base_score = float(
            json.loads(self.booster_.save_config())["learner"]["learner_model_param"][
                "base_score"
            ]
        )

        # Prepare data for binning summary
        # In newer XGBoost versions there is no need to convert base_score to logit
        scores = np.full((self.X_train.shape[0],), base_score)
        Xy_train = xgb.DMatrix(self.X_train, label=self.y_train, base_margin=scores)

        n_rounds = self.booster_.num_boosted_rounds()
        labels = Xy_train.get_label()

        df_indexes = pd.DataFrame()
        df_leafs = pd.DataFrame()
        df_binning_table = pd.DataFrame()

        # adopted from here: https://xgboost.readthedocs.io/en/latest/python/examples/individual_trees.html
        for i in range(n_rounds):
            if i == 0:
                # predict leaf index
                tree_leaf_idx = self.booster_.predict(
                    Xy_train, iteration_range=(0, i + 1), pred_leaf=True
                )
                # predict margin
                tree_leafs = self.booster_.predict(
                    Xy_train, iteration_range=(0, i + 1), output_margin=True
                ) - scores
            else:
                # Predict leaf index
                tree_leaf_idx = self.booster_.predict(
                    Xy_train, iteration_range=(0, i + 1), pred_leaf=True
                )[:, -1]
                # Predict margin
                tree_leafs = (
                    self.booster_.predict(
                        Xy_train, iteration_range=(i, i + 1), output_margin=True
                    )
                    - scores
                )

            # Get counts of events and non-events
            index_and_label = pd.concat(
                [
                    pd.Series(tree_leaf_idx, name="leaf_idx"),
                    pd.Series(labels, name="label"),
                ],
                axis=1,
            )
            # Create a binning table
            binning_table = (
                index_and_label.groupby("leaf_idx").agg(["sum", "count"]).reset_index()
            ).astype(float)
            binning_table.columns = ["leaf_idx", "Events", "Total"]
            binning_table["tree"] = i
            binning_table["NonEvents"] = (
                binning_table["Total"] - binning_table["Events"]
            )
            binning_table["EventRate"] = (
                binning_table["Events"] / binning_table["Total"]
            )
            binning_table = binning_table[
                ["tree", "leaf_idx", "Events", "NonEvents", "Total", "EventRate"]
            ]

            # Aggregate leaf indices, leafs, and counts of events and non-events
            df_indexes = pd.concat(
                [df_indexes, pd.Series(tree_leaf_idx, name=f"tree_{i}")], axis=1
            )
            df_leafs = pd.concat(
                [df_leafs, pd.Series(tree_leafs, name=f"tree_{i}")], axis=1
            )
            df_binning_table = pd.concat([df_binning_table, binning_table], axis=0)

            # Extract leaf weights (XAddEvidence)
            df_x_add_evidence = self.extract_leaf_weights()
            df_scorecard = df_x_add_evidence.merge(
                df_binning_table,
                left_on=["Tree", "Node"],
                right_on=["tree", "leaf_idx"],
                how="left",
            ).drop(["tree", "leaf_idx"], axis=1)
            df_scorecard = df_scorecard[
                [
                    "Tree",
                    "Node",
                    "Feature",
                    "Sign",
                    "Split",
                    "Events",
                    "NonEvents",
                    "Total",
                    "EventRate",
                    "XAddEvidence",
                ]
            ]
            
            # Sort by Tree and Node
            df_scorecard = df_scorecard.sort_values(by=["Tree", "Node"]).reset_index(
                drop=True
            )
            # Calculate the cumulative sum of NonEvents and Events for each tree
            df_scorecard["CumNonEvents"] = df_scorecard.groupby("Tree")[
                "NonEvents"
            ].transform("sum")
            df_scorecard["CumEvents"] = df_scorecard.groupby("Tree")[
                "Events"
            ].transform("sum")

            # Calculate Weight-of-Evidence (WOE)
            df_scorecard["WOE"] = np.log(
                (df_scorecard["NonEvents"] / df_scorecard["CumNonEvents"])
                / (df_scorecard["Events"] / df_scorecard["CumEvents"])
            ).replace([np.inf, -np.inf], 0) # type: ignore
            
            # Detailed split conditions
            if self.print_splits:
                tree_json = json_parse(self.booster_)
                output_conditions = extract_relevant_conditions(tree_json)

                # Convert to DataFrame
                output_conditions_df = pd.DataFrame.from_dict(output_conditions, orient='index', columns=['DetailedSplit']).reset_index()
                output_conditions_df.columns = ['Key', 'DetailedSplit']

                # Merge and add 'Condition' column
                df_scorecard = df_scorecard.merge(output_conditions_df[['Key', 'DetailedSplit']], left_on='XAddEvidence', right_on='Key', how='left')
                df_scorecard.drop(columns=['Key'], inplace=True)

        return df_scorecard[
            [
                "Tree",
                "Node",
                "Feature",
                "Sign",
                "Split",
                "Events",
                "NonEvents",
                "Total",
                "EventRate",
                "XAddEvidence",
                "WOE",
            ] + (["DetailedSplit"] if self.print_splits else [])
        ]


def json_parse(booster) -> List:
    ret = booster.get_dump(dump_format="json")
    return [json.loads(tree) for tree in ret]


def extract_values(obj, key):
    key_dict = {}
    info_dict = {}

    def _extract(obj, prev=None):
        nonlocal key_dict
        nonlocal info_dict

        if isinstance(obj, dict):
            try:
                info_dict[obj["nodeid"]] = {
                    "parent": prev,
                    "split_column": obj["split"],
                    "split_number": obj["split_condition"],
                    "if_less_than": obj["yes"],
                    "if_greater_than": obj["no"],
                    "if_null": obj["missing"],
                }
            except KeyError:
                info_dict[obj["nodeid"]] = {"parent": prev}
            prev = obj["nodeid"]
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    _extract(v, prev)
                elif k == key:
                    key_dict[obj["nodeid"]] = v
        elif isinstance(obj, list):
            for item in obj:
                _extract(item, prev)

    _extract(obj)

    return key_dict, info_dict


def recurse_backwards(first_node, splits) -> str:
    query_list = []

    def _recurse(x):
        nonlocal query_list
        prev_node = x
        next_node = splits[prev_node]["parent"]
        with contextlib.suppress(KeyError):
            node = splits[next_node]
            if (
                node["if_less_than"] == prev_node
                and node["if_less_than"] == node["if_null"]
            ):
                text = f"{node['split_column']} < {node['split_number']} or missing"
                query_list.insert(0, text)
                _recurse(next_node)
            elif node["if_less_than"] == prev_node:
                text = f"{node['split_column']} < {node['split_number']}"
                query_list.insert(0, text)
                _recurse(next_node)
            elif (
                node["if_greater_than"] == prev_node
                and node["if_greater_than"] == node["if_null"]
            ):
                text = f"{node['split_column']} >= {node['split_number']} or missing"
                query_list.insert(0, text)
                _recurse(next_node)
            elif node["if_greater_than"] == prev_node:
                text = f"{node['split_column']} >= {node['split_number']}"
                query_list.insert(0, text)
                _recurse(next_node)

    _recurse(first_node)

    return ", ".join(query_list)


def extract_relevant_conditions(tree_json):
    output_conditions = {}

    for tree in tree_json:
        leaves, splits = extract_values(tree, "leaf")

        if len(leaves) == 1:
            # If there is only one leaf, use its value as the key
            output_conditions[list(leaves.values())[0]] = ""
        else:
            for base_leaf in leaves:
                leaf_query = recurse_backwards(base_leaf, splits)
                output_conditions[leaves[base_leaf]] = leaf_query

    return output_conditions

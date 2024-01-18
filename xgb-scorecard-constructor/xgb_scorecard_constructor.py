import json
import numpy as np
import pandas as pd
import xgboost as xgb


class XGBScorecardConstructor:

    """
    A class for generating a scorecard from a trained XGBoost model.
    The methodology follows the NVIDIA GTC talk
    "Machine Learning in Retail Credit Risk" by Paul Edwards

    Parameters:
    - xgb_model (xgboost.XGBClassifier): The trained XGBoost model.
    - X_train (pd.DataFrame): The training data features.
    - y_train (pd.Series): The training data labels.

    Methods:
    - extract_leaf_weights: Extracts leaf weights based on the XGBoost model.
    - generate_scorecard: Generates the scorecard by combining leaf weights with binning summary.

    Example usage:
    ```
    # Instantiate the XGBScorecardConstructor
    scorecard_constructor = XGBScorecardConstructor(xgb_model, X_train, y_train)
    
    # Generate the scorecard
    xgb_scorecard = scorecard_constructor.construct_scorecard()
    
    # Print the scorecard
    print(xgb_scorecard)
    ```

    The constructed scorecard includes information such as Tree, Node, Feature, Split, Sign, events,
    non_events, total, event_rate, XAddEvidence, and WOE for each row.
    
    #TODO: add support for handling leafs with missing values

    """
    def __init__(self, xgb_model, X_train, y_train):
        self.xgb_model = xgb_model
        self.X_train = X_train
        self.y_train = y_train
        self.booster_ = xgb_model.get_booster()

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
            return condition_df[["Tree", "Node", "Feature", "Split", "Sign", "XAddEvidence"]]

        # Merge on 'Yes' and 'No' ID (True = <, False = >=)
        yes_condition_df = merge_and_rename(leaf_gains, "Yes", "<")
        no_condition_df = merge_and_rename(leaf_gains, "No", ">=")

        # Concatenate the DataFrames
        leaf_weights_df = pd.concat([yes_condition_df, no_condition_df], ignore_index=True)
        leaf_weights_df = leaf_weights_df.sort_values(by="Tree").reset_index(drop=True)

        return leaf_weights_df

    def construct_scorecard(self):
        # # retrieve base score
        base_score = float(
            json.loads(self.booster_.save_config())["learner"][
                "learner_model_param"
                ]["base_score"]
        )

        # Prepare data for binning summary
        scores = np.full((self.X_train.shape[0],), base_score)
        print(self.booster_.attr('base_score'))
        Xy_train = xgb.DMatrix(self.X_train, label=self.y_train, base_margin=scores)

        n_rounds = self.booster_.num_boosted_rounds()
        labels = Xy_train.get_label()

        df_indexes = pd.DataFrame()
        df_leafs = pd.DataFrame()
        df_binning_table = pd.DataFrame()

        for i in range(n_rounds):
            if i == 0:
                # predict leaf index
                tree_leaf_idx = self.booster_.predict(
                    Xy_train, iteration_range=(0, i + 1), pred_leaf=True
                )
                # predict margin
                tree_leafs = self.booster_.predict(
                    Xy_train, iteration_range=(0, i + 1), output_margin=True
                )

            else:
                # predict leaf index
                tree_leaf_idx = self.booster_.predict(
                    Xy_train, iteration_range=(0, i + 1), pred_leaf=True
                )[:, -1]
                # predict margin
                tree_leafs = (
                    self.booster_.predict(Xy_train, iteration_range=(i, i + 1), output_margin=True)
                    - scores
                )

            # get counts of events and non-events
            index_and_label = pd.concat(
                [
                    pd.Series(tree_leaf_idx, name="leaf_idx"),
                    pd.Series(labels, name="label"),
                ],
                axis=1,
            )

            binning_table = index_and_label.groupby("leaf_idx").agg(["sum", "count"]).reset_index()
            binning_table.columns = ['leaf_idx', 'events', 'total']
            binning_table['tree'] = i
            binning_table['non_events'] = binning_table['total'] - binning_table['events']
            binning_table['event_rate'] = binning_table['events'] / binning_table['total']
            binning_table = binning_table[['tree', 'leaf_idx', 'events', 'non_events', 'total', 'event_rate']]

            # aggregate trees
            df_indexes = pd.concat(
                [df_indexes, pd.Series(tree_leaf_idx, name=f"tree_{i}")], axis=1
            )
            df_leafs = pd.concat([df_leafs, pd.Series(tree_leafs, name=f"tree_{i}")], axis=1)
            df_binning_table = pd.concat([df_binning_table, binning_table], axis=0)

        # Extract leaf weights
        df_x_add_evidence = self.extract_leaf_weights()
        df_scorecard = df_x_add_evidence.merge(df_binning_table, left_on=['Tree', 'Node'], right_on=['tree', 'leaf_idx'], how='left').drop(['tree', 'leaf_idx'], axis=1)
        df_scorecard = df_scorecard[['Tree', 'Node', 'Feature', 'Split', 'Sign', 'events', 'non_events', 'total', 'event_rate', 'XAddEvidence']]

        # Calculate the cumulative sum of 'non_events' and 'events' for each tree
        df_scorecard['cum_non_events'] = df_scorecard.groupby('Tree')['non_events'].transform('sum')
        df_scorecard['cum_events'] = df_scorecard.groupby('Tree')['events'].transform('sum')

        # Calculate WOE for each row
        df_scorecard['WOE'] = np.log(
            (df_scorecard['non_events'] / df_scorecard['cum_non_events'])
            /
            (df_scorecard['events'] / df_scorecard['cum_events'])
        )

        return df_scorecard[['Tree', 'Node', 'Feature', 'Split', 'Sign', 'events', 'non_events', 'total', 'event_rate', 'XAddEvidence', 'WOE']]
    
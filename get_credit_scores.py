# get_credit_scores.py

import pandas as pd
import numpy as np
from typing import Dict, List

# scores for features and total score
def get_credit_scores(df: pd.DataFrame = None, features: List[str] = None,
                      scorecard: pd.DataFrame = None) -> pd.DataFrame:
    
    if df is None or features is None or scorecard is None:
        raise ValueError("df, features, and scorecard parameters are required.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a DataFrame.")

    if not isinstance(features, list):
        raise TypeError("features must be a list.")

    if not isinstance(scorecard, pd.DataFrame):
        raise TypeError("scorecard must be a DataFrame.")

    dataframe = df[features].copy()
    val_scores = np.zeros(len(dataframe))

    for feature in features:
        feature_scores = np.zeros(len(dataframe))

        for tree_id in scorecard.index.unique():
            criteria = scorecard.loc[tree_id]

            for index, row in criteria.iterrows():
                sc_feature = row['Feature']
                if sc_feature == feature:
                    sc_sign = row['Sign']
                    sc_split = row['Split']
                    sc_score = row['Points']

                    if sc_sign == '<':
                        feature_scores += (dataframe[sc_feature] < sc_split) * sc_score
                    elif sc_sign == '>=':
                        feature_scores += (dataframe[sc_feature] >= sc_split) * sc_score

        dataframe[feature] = feature_scores

    dataframe['Score'] = dataframe[features].sum(axis=1)

    return dataframe[features + ['Score']]
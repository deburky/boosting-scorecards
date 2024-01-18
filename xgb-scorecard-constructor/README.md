# XGB Scorecard Constructor

Author: Denis Burakov ([GitHub](http://github.com/deburky))

Version: `v1.0`

## Description

This Python class is designed to generate a scorecard from a trained XGBoost model. 

The methodology behind it is inspired by the NVIDIA GTC Talk "Machine Learning in Retail Credit Risk" by Paul Edwards ([GitHub](https://github.com/pedwardsada)).

## Parameters

- `xgb_model` (xgboost.XGBClassifier): The trained XGBoost model.
- `X_train` (pd.DataFrame): Features of the training data.
- `y_train` (pd.Series): Labels of the training data.

## Methods

- `extract_leaf_weights`: Extracts leaf weights based on the XGBoost model.
- `generate_scorecard`: Generates the scorecard by combining leaf weights with binning summary.

## Example Usage

```python
# Instantiate the XGBScorecardConstructor
scorecard_constructor = XGBScorecardConstructor(
    xgb_model, 
    X_train, 
    y_train
)

# Generate the scorecard
xgb_scorecard = scorecard_constructor.construct_scorecard()

# Print the scorecard
print(xgb_scorecard)

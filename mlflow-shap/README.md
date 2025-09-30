# MLflow + CatBoost + SHAP Example

This project demonstrates how to log a CatBoost model with SHAP explainer using MLflow.

## Features

- Train a CatBoost classifier on synthetic data
- Generate SHAP explanations using TreeExplainer
- Log both model and explainer to MLflow
- Demonstrate loading and using the logged artifacts
- Export feature importance and SHAP values as CSV files

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Usage

Run the example:

```bash
python main.py
```

This will:
1. Create a synthetic classification dataset
2. Train a CatBoost model with early stopping
3. Generate SHAP explanations
4. Log everything to MLflow
5. Demonstrate loading the logged model and explainer

## What Gets Logged

- **Model**: CatBoost classifier with parameters
- **SHAP Explainer**: TreeExplainer for generating SHAP values
- **Artifacts**: 
  - `shap_values.csv`: SHAP values for test samples
  - `feature_importance.csv`: Feature importance from CatBoost
- **Parameters**: Model hyperparameters
- **Metrics**: Training and test accuracy

## MLflow UI

After running the example, you can view the results in the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Key MLflow + SHAP Integration

The example shows how to:

1. **Log SHAP explainer**:
   ```python
   explainer = shap.TreeExplainer(model)
   mlflow.shap.log_explainer(explainer, artifact_path="shap_explainer")
   ```

2. **Load SHAP explainer**:
   ```python
   loaded_explainer = mlflow.shap.load_explainer(explainer_uri)
   shap_values = loaded_explainer.shap_values(X_sample)
   ```

3. **Log CatBoost model**:
   ```python
   mlflow.catboost.log_model(cb_model=model, artifact_path="model")
   ```

## Requirements

- Python >= 3.11
- CatBoost >= 1.2.8
- MLflow >= 3.4.0
- SHAP >= 0.48.0
- scikit-learn >= 1.3.0


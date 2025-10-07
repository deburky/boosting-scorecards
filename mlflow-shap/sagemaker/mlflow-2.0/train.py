#!/usr/bin/env python3
"""
train.py

Training script for SageMaker MLflow workflow with SHAP integration.

- Trains CatBoostClassifier on synthetic data
- Logs metrics, model, SHAP values, and a serializable SHAP explainer to MLflow
- Saves CatBoost model to /opt/ml/model for SageMaker deployment
"""

import argparse
import logging
import os

import mlflow
import mlflow.catboost
import mlflow.pyfunc
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from mlflow.models import infer_signature
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, feature_names


def train_catboost_model(X_train, y_train, X_val, y_val, hyperparams=None):
    """Train a CatBoost classifier."""
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    if hyperparams is None:
        hyperparams = {
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 6,
            "early_stopping_rounds": 10,
        }

    model = CatBoostClassifier(
        iterations=hyperparams.get("iterations", 100),
        learning_rate=hyperparams.get("learning_rate", 0.1),
        depth=hyperparams.get("depth", 6),
        random_seed=42,
        verbose=False,
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=hyperparams.get("early_stopping_rounds", 10),
        verbose=False,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    parser.add_argument("--l2_leaf_reg", type=float, default=5)
    parser.add_argument("--min_data_in_leaf", type=int, default=10)
    parser.add_argument("--early_stopping_rounds", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("MLflow + CatBoost + SHAP Example in SageMaker")

    # Generate synthetic dataset
    X, y, feature_names = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    logger.info("Training set: %s", X_train.shape)
    logger.info("Validation set: %s", X_val.shape)
    logger.info("Test set: %s", X_test.shape)

    # Train CatBoost
    hyperparams = {
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    model = train_catboost_model(X_train, y_train, X_val, y_val, hyperparams)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    logger.info("Training accuracy: %.4f", train_acc)
    logger.info("Test accuracy: %.4f", test_acc)

    # Set tracking URI
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("mlflow-catboost-shap")

    # MLflow logging
    with mlflow.start_run(run_name="catboost_shap_model_run"):
        mlflow.log_params(
            {
                "iterations": hyperparams["iterations"],
                "learning_rate": hyperparams["learning_rate"],
                "depth": hyperparams["depth"],
            }
        )
        signature = infer_signature(X_test, model.predict(X_test))

        # Log CatBoost model as artifact
        mlflow.catboost.log_model(model, "catboost_model")

        # Save SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_path = os.path.join(args.output_data_dir, "shap_values.csv")
        shap_df.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path)

        # Serialize SHAP explainer as custom pyfunc model
        logger.info("Creating and saving custom SHAP explainer...")

        class SerializableSHAPExplainer:
            """Custom SHAP explainer that can be serialized by MLflow."""

            def __init__(self, model):
                self.model = model
                self.explainer = shap.TreeExplainer(model)

            def predict(self, X):
                """Generate SHAP values for input data."""
                return self.explainer.shap_values(X)

            def __call__(self, X):
                return self.predict(X)

        shap_explainer = SerializableSHAPExplainer(model)

        mlflow.pyfunc.log_model(
            artifact_path="catboost_shap_explainer",  # name in MLflow 3.0
            python_model=shap_explainer,
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name="catboost-shap-explainer",
            metadata={
                "model_type": "SHAP_Explainer",
                "base_model": "CatBoostClassifier",
            },
        )

        logger.info("SHAP explainer saved successfully!")

        # Metrics
        mlflow.log_metrics({"train_acc": train_acc, "test_acc": test_acc})

    # Save for SageMaker deployment
    model.save_model(os.path.join(args.model_dir, "catboost_model"))
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train.py

Generic blueprint for model training step in SageMaker Pipeline.
This script demonstrates how to:
- Load pre-split training and test data from S3
- Train a machine learning model (example: CatBoost classifier)
- Log metrics, parameters, and artifacts to MLflow
- Save model artifacts for SageMaker deployment
- Generate model explanations (e.g., SHAP values)

Model Saving Strategy:
1. MLflow Tracking: mlflow.catboost.log_model() saves the model to MLflow's
   artifact store (S3) with model.cb, MLmodel, and environment files
2. SageMaker Deployment: model.save_model() saves to /opt/ml/model for
   SageMaker's model.tar.gz artifact (used by inference step)

Customize this template by:
1. Replacing the model type (CatBoost, XGBoost, LightGBM, etc.)
2. Updating feature definitions to match your dataset
3. Adjusting hyperparameters and evaluation metrics
4. Adding custom model artifacts and visualizations
"""

import argparse
import json
import logging
import os

import awswrangler as wr
import mlflow
import mlflow.catboost
import mlflow.data
import pandas as pd
import sagemaker_mlflow
from catboost import CatBoostClassifier, Pool
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature
from pandas.core.frame import DataFrame
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("MLflow version: %s", mlflow.__version__)
logger.info("Sagemaker MLflow version: %s", sagemaker_mlflow.__version__)

# ============================================================================
# Feature Definitions
# ============================================================================
# Customize these for your specific dataset
LABEL = "target"

NUM_FEATURES = [
    "feature_1",
    "feature_2",
    "feature_3",
]

CAT_FEATURES = [
    "category_1",
    "category_2",
]

TEXT_FEATURES = ["text_feature_1"]  # Optional text features for CatBoost

FEATURES = NUM_FEATURES + CAT_FEATURES + TEXT_FEATURES


# ============================================================================
# Helper Functions
# ============================================================================
def write_mlflow_metadata(
    run_id: str, model_id: str, output_dir: str = "/opt/ml/output/data"
):
    """
    Write MLflow metadata for pipeline propagation.
    This allows downstream steps to access the model and run information.

    Args:
        run_id: MLflow run ID
        model_id: MLflow model ID
        output_dir: Directory to write metadata file
    """
    metadata = {"mlflow_run_id": run_id, "mlflow_model_id": model_id}
    path = os.path.join(output_dir, "mlflow.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    logger.info("Saved MLflow metadata to %s", path)


def load_data_from_path(data_path: str, label: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from S3 or local path.

    Args:
        data_path: Path to the data directory (S3 or local)
        label: Name of the label column

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is labels Series
    """
    logger.info("Loading data from %s", data_path)

    try:
        # Try local storage first
        if os.path.exists(data_path):
            logger.info("Loading data from local path: %s", data_path)
            df_chunks = []
            for filename in os.listdir(data_path):
                if filename.endswith(".parquet"):
                    file_path = os.path.join(data_path, filename)
                    df_chunk = pd.read_parquet(file_path)
                    df_chunks.append(df_chunk)

            if not df_chunks:
                raise FileNotFoundError(f"No parquet files found in {data_path}")

            df = pd.concat(df_chunks)
            logger.info(
                "Successfully loaded data with %d rows and %d columns",
                df.shape[0],
                df.shape[1],
            )
        elif data_path.startswith("s3://"):
            logger.info("Loading data from S3: %s", data_path)
            df_chunks = wr.s3.read_parquet(
                data_path,
                use_threads=True,
                chunked=True,
                ignore_empty=True,
            )
            df = pd.concat(list[DataFrame](df_chunks))
            logger.info("Successfully loaded data from S3 with %d rows", df.shape[0])
        else:
            raise FileNotFoundError(f"Invalid data path: {data_path}")

        # Extract features and label
        y = df[label].copy()
        X = df.drop(columns=[label])

        return X, y

    except Exception as e:
        logger.error("Failed to load data from %s: %s", data_path, str(e))
        raise


def load_split_data(base_path: str, label: str):
    """
    Load train and test data.
    Works for both SageMaker (separate train/test channels) and local structures.

    Args:
        base_path: Base path to data
        label: Name of label column

    Returns:
        Dictionary with loaded datasets (X_train, y_train, X_test, y_test)
    """
    result = {}

    # Detect SageMaker-provided paths if running in container
    sm_train = os.environ.get("SM_CHANNEL_TRAIN")
    sm_test = os.environ.get("SM_CHANNEL_TEST")

    if sm_train and sm_test:
        train_path = sm_train
        test_path = sm_test
        logger.info(
            "Detected SageMaker environment - train: %s, test: %s",
            train_path,
            test_path,
        )
    else:
        # Local or S3 mode
        base_path = base_path.rstrip("/")
        train_path = f"{base_path}/train"
        test_path = f"{base_path}/test"
        logger.info("Loading data from base path: %s", base_path)

    # Load training dataset
    logger.info("Loading training data from %s", train_path)
    result["X_train"], result["y_train"] = load_data_from_path(train_path, label)

    # Load test dataset
    logger.info("Loading test data from %s", test_path)
    result["X_test"], result["y_test"] = load_data_from_path(test_path, label)

    logger.info("✅ Successfully loaded train/test datasets")
    return result


# ============================================================================
# Model Training Functions
# ============================================================================
def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    hyperparams=None,
    cat_features=None,
    text_features=None,
):
    """
    Train a machine learning model (example: CatBoost classifier).

    Replace this function with your preferred model training logic:
    - XGBoost: xgb.XGBClassifier()
    - LightGBM: lgb.LGBMClassifier()
    - Scikit-learn: RandomForestClassifier(), etc.
    - PyTorch/TensorFlow: Custom neural networks

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        hyperparams: Model hyperparameters
        cat_features: Categorical feature names
        text_features: Text feature names

    Returns:
        Trained model
    """
    logger.info("Creating Pool objects for training")

    # CatBoost-specific data structures
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_features,
        text_features=text_features,
    )

    val_pool = Pool(
        data=X_val,
        label=y_val,
        cat_features=cat_features,
        text_features=text_features,
    )

    if hyperparams is None:
        hyperparams = {
            "iterations": 500,
            "learning_rate": 0.1,
            "depth": 6,
            "early_stopping_rounds": 50,
        }

    logger.info("Training model with hyperparams: %s", hyperparams)

    # Initialize model (example: CatBoost)
    model = CatBoostClassifier(
        iterations=hyperparams.get("iterations", 500),
        learning_rate=hyperparams.get("learning_rate", 0.1),
        depth=hyperparams.get("depth", 6),
        l2_leaf_reg=hyperparams.get("l2_leaf_reg", 3.0),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
    )

    # Train model
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=hyperparams.get("early_stopping_rounds", 50),
        verbose=100,
    )

    # Get feature importance
    feature_importances = model.get_feature_importance(train_pool)
    feature_names = model.feature_names_
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    logger.info("\nTop 10 important features:")
    for i, (feature, importance) in enumerate(
        zip(
            feature_importance_df["Feature"].head(10),
            feature_importance_df["Importance"].head(10),
            strict=False,
        )
    ):
        logger.info("%d. %s: %.4f", i + 1, feature, importance)

    return model


# ============================================================================
# Main Function
# ============================================================================
def main():  # sourcery skip: extract-duplicate-method
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
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "s3://my-bucket/data"),
        help="Base directory containing prepared train and test data",
    )

    # Hyperparameters
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)

    # Model metadata
    parser.add_argument("--model_version", type=str, default="v1.0")

    args = parser.parse_args()

    os.makedirs(args.output_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("MLflow + ML Model Training in SageMaker")

    # Use global feature definitions
    label = LABEL
    num_features = NUM_FEATURES
    cat_features = CAT_FEATURES
    text_features = TEXT_FEATURES
    features = FEATURES

    # Load pre-split data
    data = load_split_data(args.input_data_dir, label)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Ensure correct dtypes
    X_train[num_features] = X_train[num_features].astype(float)
    X_test[num_features] = X_test[num_features].astype(float)

    for col in cat_features + text_features:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Train model
    hyperparams = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "early_stopping_rounds": args.early_stopping_rounds,
    }

    logger.info("Training with %d rows", len(X_train))
    logger.info("Testing with %d rows", len(X_test))

    model = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        hyperparams=hyperparams,
        cat_features=cat_features,
        text_features=text_features,
    )

    # Evaluate model
    logger.info("Evaluating model performance")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    gini = 2 * roc_auc - 1

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    logger.info("Training accuracy: %.4f", train_acc)
    logger.info("Test accuracy: %.4f", test_acc)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall: %.4f", recall)
    logger.info("F1 Score: %.4f", f1)
    logger.info("ROC AUC: %.4f", roc_auc)
    logger.info("Gini: %.4f", gini)

    # Set tracking URI and experiment
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("my-ml-experiment")

    # MLflow logging
    with mlflow.start_run(run_name=f"model_{args.model_version}"):
        # Log parameters
        mlflow.log_params(
            {
                "model_version": args.model_version,
                "iterations": hyperparams["iterations"],
                "learning_rate": hyperparams["learning_rate"],
                "depth": hyperparams["depth"],
                "early_stopping_rounds": hyperparams["early_stopping_rounds"],
                "training_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
            }
        )

        # Define model signature
        signature = infer_signature(X_test, model.predict_proba(X_test))

        # Log model to MLflow
        # This saves the model to MLflow's artifact store with:
        # - model.cb (CatBoost model file)
        # - MLmodel (metadata file)
        # - conda.yaml, requirements.txt, python_env.yaml
        logger.info("Logging model to MLflow")
        model_info = mlflow.catboost.log_model(
            cb_model=model,
            name="ml_model",
            signature=signature,
        )

        run_id = mlflow.active_run().info.run_id
        model_id = model_info.model_id

        write_mlflow_metadata(run_id, model_id, args.output_data_dir)
        logger.info(
            "MLflow metadata written for run_id=%s, model_id=%s", run_id, model_id
        )

        # Calculate and save SHAP values
        logger.info("Calculating SHAP values")
        test_pool = Pool(
            data=X_test,
            label=y_test,
            cat_features=cat_features,
            text_features=text_features,
        )

        shap_vals = model.get_feature_importance(
            data=test_pool,
            type="ShapValues",
        )

        # Save SHAP values (first 100 samples)
        shap_df = pd.DataFrame(
            shap_vals[:100, :-1],
            columns=features,
        )
        shap_df["base_value"] = shap_vals[:100, -1]

        shap_path = os.path.join(args.output_data_dir, "shap_values.csv")
        shap_df.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path)

        # Log metadata tags
        metadata = {
            "model_type": "Classifier",
            "model_version": args.model_version,
            "cat_features": ",".join(cat_features),
            "text_features": ",".join(text_features),
        }

        for key, value in metadata.items():
            mlflow.set_tag(key, value)

        # Log sample datasets for observability
        logger.info("Logging sample datasets for observability")

        train_sample = X_train.sample(n=min(20, len(X_train)), random_state=42)
        train_y_sample = y_train.loc[train_sample.index]
        train_sample[label] = train_y_sample

        test_sample = X_test.sample(n=min(20, len(X_test)), random_state=42)
        test_y_sample = y_test.loc[test_sample.index]
        test_sample[label] = test_y_sample

        try:
            # pylint: disable=no-member
            train_dataset: PandasDataset = mlflow.data.from_pandas(
                train_sample,
                name=f"train_sample_{args.model_version}",
                targets=label,
                source="S3",
            )
            test_dataset: PandasDataset = mlflow.data.from_pandas(
                test_sample,
                name=f"test_sample_{args.model_version}",
                targets=label,
                source="S3",
            )
            # pylint: enable=no-member

            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="holdout")
            logger.info("Successfully logged sample datasets")
        except Exception as e:
            logger.warning("Failed to log datasets: %s", str(e))
            raise

        # Log metrics
        mlflow.log_metrics(
            {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "gini": gini,
                "avg_precision": avg_precision,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        )

    # Save model for SageMaker deployment (tar.gz artifact)
    # Note: MLflow already saved the model when we called mlflow.catboost.log_model()
    # This is just for the SageMaker model.tar.gz artifact
    model_path = os.path.join(args.model_dir, "model.cb")
    model.save_model(model_path, format="cbm")
    logger.info("Model saved to %s for SageMaker deployment", model_path)

    # Save model metadata
    metadata = {
        "model_version": args.model_version,
        "features": features,
        "num_features": num_features,
        "cat_features": cat_features,
        "text_features": text_features,
    }

    with open(
        os.path.join(args.model_dir, "model_metadata.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train.py

Training script for SageMaker MLflow workflow with SHAP integration.

- Trains CatBoostClassifier on classification dataset
- Logs metrics, model, SHAP values, and a serializable SHAP explainer to MLflow
- Saves CatBoost model to /opt/ml/model for SageMaker deployment
"""

import argparse
import logging
import os

import boto3
import mlflow
import mlflow.catboost
import mlflow.data
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from imblearn.under_sampling import RandomUnderSampler
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Print mlflow version
import sagemaker_mlflow

logger.info("MLflow version: %s", mlflow.__version__)
logger.info("Sagemaker MLflow version: %s", sagemaker_mlflow.__version__)
logger.info("Running in a container...")


def load_data(input_data_dir):
    """Load dataset from S3 or local storage."""
    logger.info("Loading data from %s", input_data_dir)

    try:
        # Try to load from local storage first (in case data is mounted in container)
        if os.path.exists(input_data_dir):
            logger.info("Loading data from local path: %s", input_data_dir)
            df_chunks = []
            for filename in os.listdir(input_data_dir):
                if filename.endswith(".parquet"):
                    file_path = os.path.join(input_data_dir, filename)
                    df_chunk = pd.read_parquet(file_path)
                    df_chunks.append(df_chunk)

            if df_chunks:
                df = pd.concat(df_chunks)
                logger.info(
                    "Successfully loaded data with %s rows and %s columns",
                    df.shape[0],
                    df.shape[1],
                )
                return df
            else:
                logger.warning("No parquet files found in input directory")

        # Try to load from S3 if local loading failed
        if input_data_dir.startswith("s3://"):
            logger.info("Loading data from S3: %s", input_data_dir)
            import awswrangler as wr

            df_chunks = wr.s3.read_parquet(
                input_data_dir,
                use_threads=True,
                chunked=True,
                ignore_empty=True,
            )
            df = pd.concat(list(df_chunks))
            logger.info("Successfully loaded data from S3 with %s rows", df.shape[0])
            return df

        raise FileNotFoundError(f"Could not find data at {input_data_dir}")

    except Exception as e:
        logger.error("Failed to load data: %s", str(e))
        raise


def train_catboost_model(
    X_train,
    y_train,
    X_val,
    y_val,
    hyperparams=None,
    cat_features=None,
    text_features=None,
):
    """Train a CatBoost classifier."""
    logger.info("Creating Pool objects for training")
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

    logger.info("Training CatBoost model with hyperparams: %s", hyperparams)
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
        logger.info("%s. %s: %.4f", i + 1, feature, importance)

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
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_TRAIN",
            "s3:/my-s3-bucket/data/ml-dataset",
        ),
        help="Directory containing training data (local path or S3 URI)",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument(
        "--l2_leaf_reg",
        type=float,
        default=3.0,
        help="L2 regularization coefficient",
    )
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=0.1,
        help="Undersampling ratio for negative class",
    )
    parser.add_argument("--model_version", type=str, default="v1.1")
    args = parser.parse_args()

    os.makedirs(args.output_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("MLflow + CatBoost + SHAP Model Training in SageMaker")

    # Load real dataset
    df = load_data(args.input_data_dir)

    # If we're running with a small number of iterations (test mode),
    # sample the data to prevent memory issues
    if args.iterations <= 50:  # Test mode with reduced iterations
        logger.info(
            "Test mode detected (iterations=%s). Sampling data...", args.iterations
        )
        original_size = len(df)
        # Take a 5% random sample for testing
        df = df.sample(frac=0.05, random_state=42)
        logger.info("Sampled data: %s rows (5%% of %s rows)", len(df), original_size)

    # Define features and label
    label = "label"
    num_features = [
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "feature5",
        "feature6",
        "feature7",
    ]
    cat_features = [
        "feature8",
        "feature9",
        "feature10",
    ]
    text_features = ["feature11"]
    features = num_features + cat_features + text_features

    # Prepare data
    logger.info("Preparing data for training")
    y = df[label].copy()
    X = df[features].copy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Ensure correct dtypes
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train[num_features] = X_train[num_features].astype(float)
    X_test[num_features] = X_test[num_features].astype(float)

    for col in cat_features + text_features:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Apply undersampling for class imbalance
    logger.info("Applying undersampling with ratio %s", args.sampling_ratio)
    sampling_ratio = args.sampling_ratio
    class_counts = y_train.value_counts()
    logger.info("Original class distribution: %s", class_counts.to_dict())

    undersampler = RandomUnderSampler(
        sampling_strategy={
            0: int(class_counts[0] * sampling_ratio),
            1: class_counts[1],
        },
        random_state=42,
    )
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    logger.info("Training set after undersampling: %s", X_train_resampled.shape)

    # Train CatBoost
    hyperparams = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "early_stopping_rounds": args.early_stopping_rounds,
    }

    model = train_catboost_model(
        X_train_resampled,
        y_train_resampled,
        X_test,
        y_test,
        hyperparams=hyperparams,
        cat_features=cat_features,
        text_features=text_features,
    )

    # Evaluate with multiple metrics
    logger.info("Evaluating model performance")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    train_acc = accuracy_score(y_train_resampled, model.predict(X_train_resampled))
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
    logger.info("Average Precision: %.4f", avg_precision)

    logger.info("\nConfusion Matrix:")
    logger.info("True Negatives: %s", tn)
    logger.info("False Positives: %s", fp)
    logger.info("False Negatives: %s", fn)
    logger.info("True Positives: %s", tp)

    # Set tracking URI
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("clf-model-catboost")

    # MLflow logging
    with mlflow.start_run(run_name=f"clf_model_{args.model_version}"):
        # Log parameters
        mlflow.log_params(
            {
                "model_version": args.model_version,
                "iterations": hyperparams["iterations"],
                "learning_rate": hyperparams["learning_rate"],
                "depth": hyperparams["depth"],
                "early_stopping_rounds": hyperparams["early_stopping_rounds"],
                "sampling_ratio": args.sampling_ratio,
                "num_features": len(num_features),
                "cat_features": len(cat_features),
                "text_features": len(text_features),
                "training_samples": X_train_resampled.shape[0],
                "test_samples": X_test.shape[0],
            }
        )

        # Define model signature
        signature = infer_signature(X_test, model.predict_proba(X_test))

        # Log CatBoost model as artifact (without registration)
        logger.info("Logging CatBoost model to MLflow")
        mlflow.catboost.log_model(
            cb_model=model,
            name="catboost_model",
            signature=signature,
            # Skip model registration to avoid naming issues
        )

        # Calculate and save SHAP values
        logger.info("Calculating SHAP values")
        test_pool = Pool(
            data=X_test,
            label=y_test,
            cat_features=cat_features,
            text_features=text_features,
        )

        # First approach: Get SHAP values using CatBoost's built-in method
        logger.info("Calculating SHAP values using CatBoost's get_feature_importance")
        shap_vals = model.get_feature_importance(
            data=test_pool,
            # reference_data=test_pool,
            type="ShapValues",
        )

        # Convert to DataFrame and drop E[X]
        shap_df = pd.DataFrame(
            shap_vals[:100, :-1],  # Use 100 samples and drop expected value column
            columns=features,
        )

        # Add base prediction (E[X]) column
        shap_df["base_value"] = shap_vals[:100, -1]

        # Log SHAP values
        shap_path = os.path.join(args.output_data_dir, "shap_values.csv")
        shap_df.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path)

        # Additional metadata for the model
        metadata = {
            "model_type": "SHAP_Explainer",
            "base_model": "CatBoostClassifier",
            "model_version": args.model_version,
            "cat_features": ",".join(cat_features),
            "text_features": ",".join(text_features),
        }

        # Log metadata as separate tags for tracking
        for key, value in metadata.items():
            mlflow.set_tag(key, value)

        logger.info("SHAP explainer saved successfully!")

        # Log input samples for observability
        logger.info("Logging sample datasets for observability")

        # Create samples with reset indices to avoid duplicate indices issues
        train_indices = X_train_resampled.sample(
            n=min(20, len(X_train_resampled)), random_state=42
        ).index.tolist()

        train_sample = X_train_resampled.loc[train_indices].reset_index(drop=True)
        train_y_sample = y_train_resampled.loc[train_indices].reset_index(drop=True)
        train_sample[label] = train_y_sample

        test_indices = X_test.sample(
            n=min(20, len(X_test)), random_state=42
        ).index.tolist()
        test_sample = X_test.loc[test_indices].reset_index(drop=True)
        test_y_sample = y_test.loc[test_indices].reset_index(drop=True)
        test_sample[label] = test_y_sample

        # Ensure numeric features are proper float types
        for col in num_features:
            train_sample[col] = pd.to_numeric(
                train_sample[col], errors="coerce"
            ).astype(float)
            test_sample[col] = pd.to_numeric(test_sample[col], errors="coerce").astype(
                float
            )

        # Create MLflow datasets
        dataset_name = args.model_version.replace(".", "_")  # pylint: disable=no-member
        try:
            train_dataset: PandasDataset = mlflow.data.from_pandas(
                train_sample,
                name=f"train_df_{dataset_name}",
                targets=label,
                source="S3",
            )
            test_dataset: PandasDataset = mlflow.data.from_pandas(
                test_sample,
                name=f"test_df_{dataset_name}",
                targets=label,
                source="S3",
            )

            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="holdout")
            logger.info("Successfully logged training and test datasets")
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("Failed to log datasets: %s", str(e))

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

        # Calculate and store rating thresholds as model artifacts
        logger.info("Calculating rating thresholds from SHAP values")

        # Calculate SHAP values for test set
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

        # Convert to DataFrame and drop E[X]
        shap_df = pd.DataFrame(
            shap_vals[:, :-1],  # drop expected value column
            columns=features,
        )

        # Add base prediction (E[X]) column
        shap_df["base_value"] = shap_vals[:, -1]

        # Create SHAP delta score (sum of all feature contributions)
        shap_df["shap_delta"] = shap_df.drop(columns=["base_value"]).sum(axis=1)

        # Create business risk metrics (price and volume-weighted SHAP)
        # Get median values for safe division
        median_volume = X_test["asin_volume"].median()
        safe_volume = X_test["asin_volume"].replace([0, np.nan], median_volume)

        median_price = X_test["avg_price"].median()
        safe_price = X_test["avg_price"].replace([0, np.nan], median_price)

        # Calculate SHAP delta scores
        shap_df["shap_delta_score"] = (
            shap_df["shap_delta"] * safe_price.values * (1 / safe_volume.values)
        )

        # Log transform for more symmetric distribution
        shap_df["shap_log_score"] = np.sign(shap_df["shap_delta_score"]) * np.log1p(
            np.abs(shap_df["shap_delta_score"])
        )

        # Calculate thresholds from the positive SHAP scores only
        score = shap_df["shap_log_score"].fillna(0.0)
        pos_mask = score > 0
        score_pos = score[pos_mask]

        if len(score_pos) > 0:
            # Calculate percentiles
            q50 = np.percentile(score_pos, 50)  # median
            q95 = np.percentile(score_pos, 95)  # 95th percentile
            q99 = np.percentile(score_pos, 99)  # 99th percentile

            # Log thresholds
            thresholds = {
                "median_pos_score": q50,
                "p95_score": q95,
                "p99_score": q99,
            }

            mlflow.log_params(thresholds)
            logger.info("Rating thresholds: %s", thresholds)

            # Store thresholds as a JSON artifact for easy retrieval during inference
            threshold_path = os.path.join(
                args.output_data_dir, "rating_thresholds.json"
            )
            with open(threshold_path, "w", encoding="utf-8") as f:
                import json

                json.dump(thresholds, f)

            mlflow.log_artifact(threshold_path)
        else:
            logger.warning(
                "No positive SHAP scores found, could not calculate rating thresholds"
            )

        # Generate feature importance plot if matplotlib is available
        try:
            import matplotlib.pyplot as plt

            feature_importance = model.get_feature_importance(data=test_pool)
            sorted_idx = np.argsort(feature_importance)
            plt.figure(figsize=(10, 12))
            plt.barh(
                range(len(sorted_idx)),
                feature_importance[sorted_idx],
                align="center",
            )
            plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
            plt.title("CatBoost Feature Importance")
            plt.tight_layout()

            # Save and log the figure
            fig_path = os.path.join(args.output_data_dir, "feature_importance.png")
            plt.savefig(fig_path)
            mlflow.log_artifact(fig_path)
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning("Failed to generate feature importance plot: %s", e)

    # Save for SageMaker deployment
    model_path = os.path.join(args.model_dir, "catboost_model.cbm")
    model.save_model(model_path, format="cbm")
    logger.info("Model saved to %s", model_path)

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
        import json

        json.dump(metadata, f, indent=2)

    logger.info("Training complete!")

    # Optional: Upload model to S3
    try:
        bucket = "my-s3-bucket"
        key_prefix = "models/clf_model"
        key = f"{key_prefix}/clf_model_{args.model_version}.cbm"

        s3 = boto3.client("s3")
        s3.upload_file(model_path, bucket, key)
        logger.info("Model also uploaded to s3://%s/%s", bucket, key)
    except Exception as e:
        logger.warning("Failed to upload model to S3: %s", str(e))


if __name__ == "__main__":
    main()

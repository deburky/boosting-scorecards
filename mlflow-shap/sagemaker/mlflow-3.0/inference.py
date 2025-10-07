#!/usr/bin/env python3
"""
inference.py

Script for batch inference using trained CatBoost model with SHAP explanations.
"""

import argparse
import logging
import os
from datetime import datetime

import boto3
import mlflow
import pandas as pd
from catboost import CatBoostClassifier, Pool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default configurations
BUCKET = "my-s3-bucket"
MODEL_PREFIX = "models/clf_model"
S3_DATA_PATH = "s3://my-s3-bucket/data/ml-dataset"


def load_model_from_s3(bucket, key):
    """Load a CatBoost model from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key (path to model file)

    Returns:
        CatBoost model object
    """
    logger.info("Loading model from s3://%s/%s", bucket, key)

    local_model_path = os.path.join("/tmp", os.path.basename(key))

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    # Download the model
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_model_path)
    logger.info("Model downloaded to %s", local_model_path)

    # Load the model
    model = CatBoostClassifier()
    model.load_model(local_model_path)
    logger.info("Model loaded successfully")

    return model


def load_model(model_uri=None, model_version="v1.1", model_path=None):
    """Load model from MLflow or S3.

    Args:
        model_uri: MLflow model URI (if using MLflow)
        model_version: Model version string (used if model_uri is None)
        model_path: Path to local model file

    Returns:
        CatBoost model object
    """
    if model_path:
        logger.info("Loading model from local path: %s", model_path)
        model = CatBoostClassifier()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
    elif model_uri:
        logger.info("Loading model from MLflow URI: %s", model_uri)
        model = mlflow.catboost.load_model(model_uri)
    else:
        # Load from S3
        bucket = BUCKET
        key = f"{MODEL_PREFIX}/clf_model_{model_version}.cbm"
        model = load_model_from_s3(bucket, key)

    return model


def batch_predict_with_shap(
    model,
    data_path,
    batch_size=10000,
    output_path=None,
    model_version="v1.1",
    sample_size=None,
):
    """Generate predictions and SHAP explanations for large datasets in batches.

    Args:
        model: CatBoost model
        data_path: Path to data (S3 URI or local path)
        batch_size: Number of rows to process in each batch
        output_path: Path to write output (S3 URI or local path)
        model_version: Model version for tracking
        sample_size: Optional limit on number of records to process

    Returns:
        Dictionary with summary statistics
    """
    # Define features
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

    with mlflow.start_run(run_name=f"Batch_Inference_{model_version}"):
        # Start time for tracking
        start_time = datetime.now()

        # Process data in batches
        total_samples = 0
        batch_count = 0
        all_results_dfs = []

        # Log parameters
        mlflow.log_params(
            {
                "model_version": model_version,
                "batch_size": batch_size,
                "data_path": data_path,
                "output_path": output_path,
                "sample_size": sample_size,
            }
        )

        # Define reader
        reader_kwargs = {"chunksize": batch_size}
        if data_path.startswith("s3://"):
            import awswrangler as wr

            chunks = wr.s3.read_parquet(data_path, chunked=True, **reader_kwargs)
        else:
            chunks = pd.read_parquet(data_path, **reader_kwargs)

        # Track how many samples we've processed if sample_size is specified
        samples_processed = 0

        # Process each batch
        for batch_idx, df_batch in enumerate(chunks):
            # Check if we've reached the sample size limit
            if sample_size is not None:
                if samples_processed >= sample_size:
                    logger.info(
                        "Reached sample size limit of %s. Stopping.",
                        sample_size,
                    )
                    break

                # If this batch would exceed the sample size, take just what we need
                if samples_processed + len(df_batch) > sample_size:
                    remaining = sample_size - samples_processed
                    logger.info(
                        "Taking %s samples from batch to reach sample size %s",
                        remaining,
                        sample_size,
                    )
                    df_batch = df_batch.sample(n=remaining, random_state=42)

            # Make a copy to avoid modifying original data
            df_batch = df_batch.copy()
            batch_count += 1

            # Log progress
            if batch_idx % 10 == 0:
                logger.info(
                    "Processing batch %s, total samples: %s",
                    batch_idx,
                    total_samples,
                )

            # Ensure correct dtypes
            df_batch[num_features] = df_batch[num_features].astype(float)
            for col in cat_features + text_features:
                df_batch[col] = df_batch[col].astype(str)

            # Create pool for prediction
            pool = Pool(
                data=df_batch[features],
                cat_features=cat_features,
                text_features=text_features,
            )

            # Make prediction
            logger.info("Generating predictions")
            df_batch["score"] = model.predict_proba(pool)[:, 1]

            # Calculate SHAP values
            logger.info("Calculating SHAP values")
            shap_vals = model.get_feature_importance(
                pool=pool,
                type="ShapValues",
            )

            # Convert to DataFrame and drop E[X]
            shap_df = pd.DataFrame(
                shap_vals[:, :-1],  # drop expected value column
                columns=features,
            )

            # Add base prediction (E[X]) column
            shap_df["base_value"] = shap_vals[:, -1]

            # Add columns with _shap suffix
            shap_df.columns = [f"{col}_shap" for col in shap_df.columns]

            # Merge SHAP values with the original batch
            shap_df.index = df_batch.index
            df_batch = pd.concat([df_batch, shap_df], axis=1)

            total_samples += len(df_batch)
            samples_processed += len(df_batch)
            all_results_dfs.append(df_batch)

            # For extremely large datasets, save intermediate results
            if output_path and batch_count % 10 == 0:
                tmp_df = pd.concat(all_results_dfs)
                save_batch_results(tmp_df, f"{output_path}/partial_{batch_idx}")
                all_results_dfs = []  # Clear to save memory

        # Combine all results
        if all_results_dfs:
            combined_results = pd.concat(all_results_dfs)
        else:
            combined_results = pd.DataFrame()  # Empty if all were saved

        # Save final results if path provided
        if output_path and len(combined_results) > 0:
            save_batch_results(combined_results, output_path)

        # Calculate end time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Log metrics
        mlflow.log_metrics(
            {
                "total_samples": total_samples,
                "total_batches": batch_count,
                "processing_time_seconds": processing_time,
                "samples_per_second": total_samples / processing_time,
            }
        )

        logger.info("Processed %s samples in %s batches", total_samples, batch_count)
        logger.info("Processing time: %.2f seconds", processing_time)

        # Return summary
        return {
            "total_samples": total_samples,
            "total_batches": batch_count,
            "processing_time_seconds": processing_time,
            "results": None if output_path else combined_results,
        }


def save_batch_results(df, output_path):
    """Save batch processing results to S3 or local storage.

    Args:
        df: DataFrame with results
        output_path: Path to save results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine if S3 or local path
    if output_path.startswith("s3://"):
        import awswrangler as wr

        # Parse S3 path
        s3_path = output_path.rstrip("/")
        s3_path = f"{s3_path}/results_{timestamp}.parquet"
        logger.info("Saving results to S3: %s", s3_path)
        wr.s3.to_parquet(df, s3_path)
    else:
        # Local path
        os.makedirs(output_path, exist_ok=True)
        local_path = os.path.join(output_path, f"results_{timestamp}.parquet")
        logger.info("Saving results locally to: %s", local_path)
        df.to_parquet(local_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference with SHAP explanations for HRV model"
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="MLflow model URI",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.1",
        help="Model version for S3 loading",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file or tarball",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=S3_DATA_PATH,
        help="Path to data (S3 URI or local path)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of rows to process in each batch",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Maximum number of samples to process (optional)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save results (S3 URI or local path)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI",
    )

    args = parser.parse_args()

    # Set MLflow tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    # Generate default output path if not specified
    if not args.output_path:
        # Create local path
        default_output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "outputs",
            f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        args.output_path = default_output
        os.makedirs(default_output, exist_ok=True)
        logger.info("No output path specified, using: %s", default_output)

    # Load model
    model = load_model(args.model_uri, args.model_version, args.model_path)

    # Run batch inference
    results = batch_predict_with_shap(
        model=model,
        data_path=args.data_path,
        batch_size=args.batch_size,
        output_path=args.output_path,
        model_version=args.model_version,
        sample_size=args.sample_size,
    )

    # Print summary
    logger.info("\nBatch inference summary:")
    logger.info("Total samples processed: %s", results["total_samples"])
    logger.info("Total batches: %s", results["total_batches"])
    logger.info("Processing time: %.2f seconds", results["processing_time_seconds"])
    logger.info("Results saved to: %s", args.output_path)


if __name__ == "__main__":
    main()

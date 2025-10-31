#!/usr/bin/env python3
"""
inference.py

Generic blueprint for distributed batch inference step in SageMaker Pipeline.
This script demonstrates how to:
- Load a trained model from previous training step
- Perform distributed batch inference using Ray
- Process large datasets efficiently (100M+ records)
- Save predictions to S3
- Optionally run Glue crawler for data cataloging

Customize this template by:
1. Adjusting Ray configuration for your compute resources
2. Modifying prediction logic for your model type
3. Adding custom postprocessing or business logic
4. Configuring output formats and destinations
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import boto3
import pandas as pd
import ray
from catboost import CatBoostClassifier, Pool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"Ray version: {ray.__version__}")


# ============================================================================
# Model Loading Functions
# ============================================================================
def load_model_from_s3(s3_path: str) -> CatBoostClassifier:
    """
    Load a model directly from S3.

    Args:
        s3_path: S3 path to the model file

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from S3: {s3_path}")

    # Parse bucket/key
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    # Download to tmp
    local_path = f"/tmp/{os.path.basename(key)}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, key, local_path)
    logger.info(f"Model downloaded to {local_path}")

    # Load model
    model = CatBoostClassifier()
    if local_path.endswith(".cb") or local_path.endswith(".cbm"):
        model.load_model(local_path, format="cbm")
    else:
        model.load_model(local_path)

    logger.info("Model loaded successfully")
    return model


def extract_features_from_model(model: CatBoostClassifier) -> tuple:
    """
    Extract feature information from the trained model.

    Args:
        model: Loaded model

    Returns:
        Tuple of (numerical_features, categorical_features, text_features)
    """
    all_features = model.feature_names_
    if not all_features:
        raise ValueError("No feature names found in model")

    # Get categorical and text feature indices
    cat_indices = model.get_cat_feature_indices()

    try:
        text_indices = model.get_text_feature_indices()
    except (AttributeError, NotImplementedError):
        logger.warning("Text feature indices not available")
        text_indices = []

    # Create feature lists
    cat_features = [all_features[idx] for idx in cat_indices]
    text_features = [all_features[idx] for idx in text_indices] if text_indices else []
    num_features = [
        f for f in all_features if f not in cat_features and f not in text_features
    ]

    logger.info(
        f"Features extracted: {len(num_features)} numeric, "
        f"{len(cat_features)} categorical, {len(text_features)} text"
    )

    return num_features, cat_features, text_features


# ============================================================================
# Ray Actor for Distributed Processing
# ============================================================================
@ray.remote
class ModelActor:
    """
    Ray Actor that holds a model instance for parallel processing.
    Each worker will have its own copy of the model in memory.
    """

    def __init__(self, model_path: str):
        self.model = None

        try:
            import uuid

            worker_id = str(uuid.uuid4())[:8]
            logger.info(
                f"ModelActor {worker_id} initializing with model from {model_path}"
            )

            # Load model
            if model_path.startswith("s3://"):
                # Download from S3
                parts = model_path.replace("s3://", "").split("/", 1)
                bucket, key = parts[0], parts[1]
                local_path = f"/tmp/model_{worker_id}_{os.path.basename(key)}"

                s3_client = boto3.client("s3")
                s3_client.download_file(bucket, key, local_path)
                logger.info(f"ModelActor {worker_id} downloaded model")

                self.model = CatBoostClassifier()
                self.model.load_model(local_path, format="cbm")
            else:
                # Local file
                self.model = CatBoostClassifier()
                self.model.load_model(model_path, format="cbm")

            # Extract features
            self.features = self.model.feature_names_
            cat_indices = self.model.get_cat_feature_indices()

            try:
                text_indices = self.model.get_text_feature_indices()
            except (AttributeError, NotImplementedError):
                text_indices = []

            self.cat_features = [self.features[idx] for idx in cat_indices]
            self.text_features = (
                [self.features[idx] for idx in text_indices] if text_indices else []
            )
            self.num_features = [
                f
                for f in self.features
                if f not in self.cat_features and f not in self.text_features
            ]

            logger.info(f"ModelActor {worker_id} initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing ModelActor: {str(e)}", exc_info=True)
            raise

    def get_features(self):
        return self.features

    def get_cat_features(self):
        return self.cat_features

    def get_text_features(self):
        return self.text_features

    def get_num_features(self):
        return self.num_features

    def predict_batch(self, batch_data: pd.DataFrame):
        """
        Predict on a batch of data.

        Args:
            batch_data: DataFrame with features

        Returns:
            Tuple of (predictions, shap_values, features)
        """
        pool = Pool(
            data=batch_data[self.features],
            cat_features=self.cat_features,
            text_features=self.text_features,
        )

        # Get predictions
        scores = self.model.predict_proba(pool)[:, 1]

        # Get SHAP values (optional - for explainability)
        shap_vals = self.model.get_feature_importance(data=pool, type="ShapValues")

        return scores, shap_vals, self.features


# ============================================================================
# Batch Processing Function
# ============================================================================
@ray.remote
def process_batch(
    batch_id: int,
    data_shard: pd.DataFrame,
    model_ref,
) -> pd.DataFrame:
    """
    Process a batch of data with the model actor.

    Args:
        batch_id: Batch identifier
        data_shard: DataFrame shard to process
        model_ref: Ray reference to ModelActor

    Returns:
        DataFrame with predictions
    """
    batch_start = time.time()
    logger.info(f"Starting batch {batch_id} with {len(data_shard)} rows")

    df_batch = data_shard.copy()

    # Get model features
    cat_features = ray.get(model_ref.get_cat_features.remote())
    text_features = ray.get(model_ref.get_text_features.remote())
    num_features = ray.get(model_ref.get_num_features.remote())

    # Prepare data
    df_batch[num_features] = df_batch[num_features].fillna(0).astype(float)
    for col in cat_features + text_features:
        df_batch[col] = df_batch[col].fillna("").astype(str)

    # Get predictions and SHAP values
    scores, shap_vals, used_features = ray.get(model_ref.predict_batch.remote(df_batch))

    df_batch["prediction_score"] = scores

    # Add SHAP values as separate columns (optional)
    shap_df = pd.DataFrame(shap_vals[:, :-1], columns=used_features)
    shap_df["base_value"] = shap_vals[:, -1]
    shap_df.columns = [f"{c}_shap" for c in shap_df.columns]
    shap_df.index = df_batch.index

    df_batch = pd.concat([df_batch, shap_df], axis=1)

    batch_time = time.time() - batch_start
    logger.info(f"Completed batch {batch_id} in {batch_time:.2f} seconds")

    return df_batch


# ============================================================================
# Distributed Inference Function
# ============================================================================
def run_distributed_inference(
    model_path: str,
    data_path: str,
    output_path: str,
    batch_size: int = 10000,
    num_cpus: int = None,
) -> str:
    """
    Run inference using Ray for distributed processing.

    Args:
        model_path: Path to model file (S3 or local)
        data_path: Path to input data (S3 or local)
        output_path: Path for output results (S3 or local)
        batch_size: Number of rows per batch
        num_cpus: Number of CPUs for Ray to use

    Returns:
        Path where results were saved
    """
    # Configure Ray
    ray_config = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "_temp_dir": "/tmp/ray",
        "logging_level": "info",
    }

    if num_cpus:
        ray_config["num_cpus"] = num_cpus

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(**ray_config)
        logger.info("Ray initialized successfully")

    logger.info(f"Ray resources: {ray.cluster_resources()}")

    start_time = time.time()

    # Create model actor
    logger.info(f"Loading model in Ray actor from {model_path}")
    model_actor = ModelActor.remote(model_path)

    # Load data with Ray
    import ray.data

    if data_path.startswith("s3://"):
        logger.info(f"Loading S3 dataset from {data_path}")
        ds = ray.data.read_parquet(data_path)
    else:
        logger.info(f"Loading local dataset from {data_path}")
        if os.path.isdir(data_path):
            file_list = [
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith(".parquet") or f.endswith(".pq")
            ]
            ds = ray.data.read_parquet(file_list)
        else:
            ds = ray.data.read_parquet(data_path)

    total_rows = ds.count()
    logger.info(f"Dataset loaded with {total_rows} rows")
    num_batches = (total_rows + batch_size - 1) // batch_size
    logger.info(f"Processing in ~{num_batches} batches of {batch_size}")

    # Process batches
    pending = []
    processed_rows = 0
    chunk_count = 0
    chunk_paths = []

    chunk_dir = (
        output_path if not output_path.startswith("s3://") else f"{output_path}/chunks"
    )
    if not output_path.startswith("s3://"):
        os.makedirs(chunk_dir, exist_ok=True)

    for i, batch in enumerate(ds.iter_batches(batch_size=batch_size)):
        if i % 10 == 0:
            logger.info(f"Submitting batch {i + 1}/{num_batches}")

        df_batch = pd.DataFrame(batch)
        obj = process_batch.remote(i, df_batch, model_actor)
        pending.append((i, obj))

        # Flush every 10 batches
        if (i % 10 == 9) or (i == num_batches - 1):
            obj_refs = [r for (_, r) in pending]
            logger.info(f"Waiting for {len(obj_refs)} pending batch results")

            while obj_refs:
                ready, obj_refs = ray.wait(obj_refs, num_returns=min(len(obj_refs), 8))
                ready_set = set(ready)
                completed = [(bid, ref) for (bid, ref) in pending if ref in ready_set]

                try:
                    batch_dfs = ray.get([ref for (_, ref) in completed])
                    chunk_results = [
                        df for df in batch_dfs if df is not None and len(df) > 0
                    ]
                except Exception as e:
                    logger.error(f"Error retrieving batch results: {str(e)}")
                    chunk_results = []

                pending = [(bid, ref) for (bid, ref) in pending if ref not in ready_set]

                elapsed = time.time() - start_time
                processed_rows += sum(len(df) for df in chunk_results)
                rps = processed_rows / elapsed if elapsed > 0 else 0

                if chunk_results:
                    try:
                        chunk_df = pd.concat(chunk_results, axis=0)

                        import random

                        random_suffix = random.randint(1000, 9999)
                        chunk_path = f"{chunk_dir}/chunk_{chunk_count:04d}_{random_suffix}.parquet"

                        logger.info(
                            f"Saving chunk {chunk_count} with {len(chunk_df)} rows"
                        )

                        if not output_path.startswith("s3://"):
                            chunk_df.to_parquet(chunk_path, index=False)
                        else:
                            import awswrangler as wr

                            wr.s3.to_parquet(
                                df=chunk_df,
                                path=chunk_path,
                                index=False,
                                dataset=False,
                            )

                        chunk_paths.append(chunk_path)
                        chunk_count += 1
                    except Exception as e:
                        logger.error(
                            f"Error saving chunk {chunk_count}: {e}", exc_info=True
                        )

                logger.info(
                    f"Progress: {processed_rows}/{total_rows} rows "
                    f"({processed_rows / total_rows:.1%}) in {elapsed:.1f}s "
                    f"({rps:.1f} rows/s)"
                )

    if chunk_paths:
        logger.info(f"Saved {chunk_count} chunks with a total of {processed_rows} rows")
        return output_path
    else:
        logger.error("No results to save")
        return None


# ============================================================================
# Glue Crawler Function (Optional)
# ============================================================================
def run_glue_crawler(crawler_name: str, region: str, target_path: str):
    """
    Run AWS Glue crawler to catalog the inference results.

    Args:
        crawler_name: Name of the Glue crawler
        region: AWS region
        target_path: S3 path to crawl
    """
    logger.info(f"Running Glue crawler {crawler_name} on {target_path}")

    glue_client = boto3.client("glue", region_name=region)

    try:
        # Start crawler
        glue_client.start_crawler(Name=crawler_name)
        logger.info(f"Started Glue crawler {crawler_name}")

        # Wait for completion
        status = "RUNNING"
        while status in ["RUNNING", "STOPPING"]:
            time.sleep(30)
            crawler_info = glue_client.get_crawler(Name=crawler_name)
            status = crawler_info["Crawler"]["State"]
            logger.info(f"Crawler status: {status}")

        logger.info(f"Glue crawler completed with status: {status}")

    except Exception as e:
        logger.error(f"Error running Glue crawler: {str(e)}")


# ============================================================================
# Main Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Distributed batch inference using Ray"
    )

    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to input data"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to model file"
    )
    parser.add_argument(
        "--model-version", type=str, default="v1.0", help="Model version"
    )
    parser.add_argument(
        "--output-path", type=str, default=None, help="Output path for results"
    )
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size")
    parser.add_argument(
        "--num-cpus", type=int, default=None, help="Number of CPUs for Ray"
    )
    parser.add_argument("--region", type=str, required=True, help="AWS region")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument(
        "--glue-crawler", type=str, default=None, help="Glue crawler name"
    )
    parser.add_argument(
        "--glue-database", type=str, default=None, help="Glue database name"
    )
    parser.add_argument("--glue-table", type=str, default=None, help="Glue table name")
    parser.add_argument(
        "--mlflow-metadata", type=str, default=None, help="Path to MLflow metadata"
    )

    args = parser.parse_args()

    # Load MLflow metadata if provided
    if args.mlflow_metadata:
        if meta_files := [
            f for f in os.listdir(args.mlflow_metadata) if f.endswith(".json")
        ]:
            with open(
                os.path.join(args.mlflow_metadata, meta_files[0]), encoding="utf-8"
            ) as f:
                meta = json.load(f)
            logger.info(f"Loaded MLflow metadata: {meta}")

    # Set output path
    if not args.output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"s3://{args.bucket}/outputs/inference/{timestamp}"

    logger.info("Starting distributed inference")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output path: {args.output_path}")

    # Run inference
    output_path = run_distributed_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_cpus=args.num_cpus,
    )

    logger.info(f"Inference complete. Results saved to {output_path}")

    # Run Glue crawler if specified
    if args.glue_crawler and output_path and output_path.startswith("s3://"):
        run_glue_crawler(args.glue_crawler, args.region, output_path)

    # Save metadata for next step
    output_metadata = {
        "s3_output_path": output_path,
        "glue_table": args.glue_table,
        "glue_database": args.glue_database,
    }

    os.makedirs("/opt/ml/processing/output_metadata", exist_ok=True)
    with open("/opt/ml/processing/output_metadata/metadata.json", "w") as f:
        json.dump(output_metadata, f)

    logger.info("✅ Inference metadata saved")


if __name__ == "__main__":
    main()

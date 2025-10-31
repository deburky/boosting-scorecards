#!/usr/bin/env python3
"""
data_preparation.py

Script for preparing training data for the ML model:
- Optionally runs a Glue job to generate the initial dataset
- Loads data from the generated dataset
- Performs train-test split and feature preparation
- Applies undersampling for class balance (optional)
- Saves train and test datasets to S3 for model training
"""

import argparse
import json
import logging
import time
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Label column
LABEL = "is_default"

# Numeric features
NUM_FEATURES = [
    "average_balance",
    "past_delinquency",
]

# Categorical features
CAT_FEATURES = [
    "product_type",
    "acquisition_channel",
]

# Text features
TEXT_FEATURES = ["customer_support_logs"]

# Combined features list
FEATURES = NUM_FEATURES + CAT_FEATURES + TEXT_FEATURES


def run_glue_job(
    job_args: dict,
    region: str = "us-east-1",
    wait: bool = True,
    timeout: int = 3600,
) -> str:
    """
    Run the ETL job in AWS Glue with the specified arguments.

    Args:
        job_args: Dictionary of arguments to pass to the Glue job
        region: AWS region
        wait: Whether to wait for the job to complete
        timeout: Maximum time to wait for job completion (in seconds)

    Returns:
        Job ID of the started Glue job
    """
    # Fixed job name based on existing convention
    job_name = "etl-feature-engineering"
    logger.info(f"Starting Glue job {job_name} with arguments: {job_args}")

    # Initialize Glue client
    glue_client = boto3.client("glue", region_name=region)

    logger.info(f"Glue job arguments: {job_args}")

    # Start the Glue job
    response = glue_client.start_job_run(
        JobName=job_name,
        Arguments=job_args,
    )

    job_id = response["JobRunId"]
    logger.info(f"Started Glue job with run ID: {job_id}")

    if wait:
        wait_for_glue_job(glue_client, job_name, job_id, timeout)

    return job_id


def wait_for_glue_job(
    glue_client, job_name: str, job_id: str, timeout: int = 3600
) -> str:
    """
    Wait for a Glue job to complete.

    Args:
        glue_client: Boto3 Glue client
        job_name: Name of the Glue job
        job_id: Job run ID
        timeout: Maximum time to wait (in seconds)

    Returns:
        Final status of the job
    """
    logger.info(f"Waiting for Glue job {job_name} (run ID: {job_id}) to complete")

    status = "STARTING"
    start_time = time.time()

    while (
        status in ["STARTING", "RUNNING", "WAITING"]
        and (time.time() - start_time) < timeout
    ):
        response = glue_client.get_job_run(JobName=job_name, RunId=job_id)
        status = response["JobRun"]["JobRunState"]

        logger.info(f"Glue job status: {status}")

        if status in ["STARTING", "RUNNING", "WAITING"]:
            time.sleep(30)  # Check every 30 seconds

    elapsed_time = time.time() - start_time

    if status == "SUCCEEDED":
        logger.info(f"Glue job completed successfully in {elapsed_time:.1f} seconds")
    elif (time.time() - start_time) >= timeout:
        logger.warning(
            f"Timeout waiting for Glue job completion after {elapsed_time:.1f} seconds"
        )
    elif status == "FAILED":
        logger.warning(
            f"Glue job failed with status: {status}. Will use default dataset."
        )
    else:
        logger.info(f"Glue job ended with status: {status}")

    return status


def check_glue_output(output_path: str, region: str = "us-east-1") -> bool:
    """
    Check if Glue job output exists.

    Args:
        output_path: S3 path where the Glue job should have written data
        region: AWS region

    Returns:
        True if output exists, False otherwise
    """
    logger.info(f"Checking for Glue job output at {output_path}")

    if not output_path.startswith("s3://"):
        raise ValueError(f"Output path must be an S3 path, got: {output_path}")

    # Parse bucket and prefix from output path
    bucket = output_path.split("/")[2]
    prefix = "/".join(output_path.split("/")[3:])
    if not prefix.endswith("/"):
        prefix += "/"

    s3_client = boto3.client("s3", region_name=region)

    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=1,
    )

    if "Contents" in response:
        logger.info(f"Found output at {output_path}")
        return True
    else:
        logger.warning(f"No output found at {output_path}")
        return False


def load_data(input_data_dir: str, sample_size: int | None = None) -> pd.DataFrame:
    """
    Load dataset from S3 or local storage with optional initial sampling.

    Args:
        input_data_dir: S3 path or local directory containing parquet files
        sample_size: Optional initial sample size to reduce memory footprint

    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from {input_data_dir}")

    try:
        # Try loading from S3 first
        if input_data_dir.startswith("s3://"):
            logger.info(f"Loading data from S3: {input_data_dir}")
            import awswrangler as wr

            # If sample_size is provided, use it to limit initial data load
            if sample_size:
                logger.info(f"Using initial sample size of {sample_size}")
                # Get list of parquet files
                objects = wr.s3.list_objects(input_data_dir)

                # Calculate approx number of files needed based on average file size
                total_objects = len(objects)
                sample_fraction = min(
                    1.0, 2 * sample_size / 134000000
                )  # 2x safety factor
                files_to_read = max(1, int(total_objects * sample_fraction))

                logger.info(
                    f"Reading {files_to_read} of {total_objects} files for initial sampling"
                )
                df = wr.s3.read_parquet(
                    path=objects[:files_to_read],
                    use_threads=True,
                )

                # Further sample if needed
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
            else:
                # Load all data
                df_chunks = wr.s3.read_parquet(
                    path=input_data_dir,
                    use_threads=True,
                    chunked=True,
                    ignore_empty=True,
                )
                df = pd.concat(list(df_chunks))

            logger.info(f"Successfully loaded {len(df)} rows from S3")
            return df

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    apply_undersampling: bool = False,
    sampling_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare train and test sets with proper feature handling.

    Args:
        df: Input DataFrame
        test_size: Proportion of data to use for test set
        random_state: Random seed
        apply_undersampling: Whether to apply undersampling to balance classes
        sampling_ratio: Ratio of negative to positive samples

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Use the global feature definitions from the top of the file
    label = LABEL
    num_features = NUM_FEATURES
    cat_features = CAT_FEATURES
    text_features = TEXT_FEATURES
    features = FEATURES

    # Prepare data
    logger.info("Preparing data for train-test split")

    # Check that required columns exist
    for col in [label] + features:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in DataFrame")

    y = df[label].copy()
    X = df[features].copy()

    # Split data
    logger.info(
        f"Splitting data with test_size={test_size}, random_state={random_state}"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Ensure correct dtypes
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train[num_features] = X_train[num_features].astype(float)
    X_test[num_features] = X_test[num_features].astype(float)

    for col in cat_features + text_features:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Apply undersampling for class imbalance if requested
    if apply_undersampling:
        logger.info(f"Applying undersampling with ratio {sampling_ratio}")
        class_counts = y_train.value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")

        undersampler = RandomUnderSampler(
            sampling_strategy={
                0: int(class_counts[0] * sampling_ratio),
                1: class_counts[1],
            },
            random_state=random_state,
        )
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        logger.info(f"Training set after undersampling: {X_train.shape}")

    # Log class distribution in train and test sets
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)

    logger.info(f"Train set size: {len(X_train)} rows")
    logger.info(f"Test set size: {len(X_test)} rows")
    logger.info(f"Train class distribution: {train_dist.to_dict()}")
    logger.info(f"Test class distribution: {test_dist.to_dict()}")

    # Store feature metadata
    feature_metadata = {
        "num_features": num_features,
        "cat_features": cat_features,
        "text_features": text_features,
        "label": label,
    }

    # Add feature metadata to the return values
    return X_train, X_test, y_train, y_test, feature_metadata


def save_metadata(
    metadata: dict,
    output_bucket: str,
    output_prefix: str,
) -> str:
    """
    Save metadata about the dataset preparation process to S3.

    Args:
        metadata: Dictionary containing metadata
        output_bucket: S3 bucket name
        output_prefix: S3 key prefix

    Returns:
        S3 path to the metadata file
    """
    logger.info(f"Saving metadata to s3://{output_bucket}/{output_prefix}")

    from datetime import datetime

    import numpy as np

    # Add timestamp to metadata
    metadata["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert NumPy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {
                convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert any NumPy types in the metadata to Python native types
    metadata_converted = convert_numpy_types(metadata)

    # Save metadata to S3
    metadata_path = f"s3://{output_bucket}/{output_prefix}/metadata.json"

    # Write metadata to S3 using boto3 directly since we have a dictionary, not a DataFrame
    s3_client = boto3.client("s3")

    # Convert dictionary to JSON string
    metadata_json = json.dumps(metadata_converted, indent=2)

    # Extract bucket and key from S3 path
    bucket_name = output_bucket
    key = f"{output_prefix.rstrip('/')}/metadata.json"

    # Upload JSON string as bytes
    s3_client.put_object(
        Body=metadata_json.encode("utf-8"),
        Bucket=bucket_name,
        Key=key,
        ContentType="application/json",
    )

    logger.info(f"Metadata saved to {metadata_path}")

    return metadata_path


def save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_bucket: str,
    output_prefix: str,
    metadata: dict = None,
    original_train_df: pd.DataFrame = None,
) -> str:
    """
    Save train and test datasets to S3 along with metadata.

    Args:
        train_df: Training DataFrame (resampled if undersampling was applied)
        test_df: Testing DataFrame
        output_bucket: S3 bucket name
        output_prefix: S3 key prefix
        metadata: Optional dictionary containing metadata about the dataset creation process
        original_train_df: Optional original training DataFrame before undersampling

    Returns:
        S3 base path where data was saved
    """
    logger.info(f"Saving datasets to s3://{output_bucket}/{output_prefix}")

    import awswrangler as wr

    # Save original training data if provided (before undersampling)
    if original_train_df is not None:
        # FIX: Save original training data to train/ directory
        train_orig_path = f"s3://{output_bucket}/{output_prefix}/train/"
        wr.s3.to_parquet(
            df=original_train_df, path=train_orig_path, dataset=True, mode="overwrite"
        )
        logger.info(
            f"Saved {len(original_train_df)} original training samples to {train_orig_path}"
        )

        # Save resampled training data separately
        train_resampled_path = f"s3://{output_bucket}/{output_prefix}/train-resampled/"
        wr.s3.to_parquet(
            df=train_df, path=train_resampled_path, dataset=True, mode="overwrite"
        )
        logger.info(
            f"Saved {len(train_df)} resampled training samples to {train_resampled_path}"
        )
    else:
        # If no original training data provided, save as usual
        train_path = f"s3://{output_bucket}/{output_prefix}/train/"
        wr.s3.to_parquet(df=train_df, path=train_path, dataset=True, mode="overwrite")
        logger.info(f"Saved {len(train_df)} training samples to {train_path}")

    # Save test dataset
    test_path = f"s3://{output_bucket}/{output_prefix}/test/"
    wr.s3.to_parquet(df=test_df, path=test_path, dataset=True, mode="overwrite")
    logger.info(f"Saved {len(test_df)} test samples to {test_path}")

    # Save metadata if provided
    if metadata:
        save_metadata(metadata, output_bucket, output_prefix)

    # Return the base path
    return f"s3://{output_bucket}/{output_prefix}/"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for the ML model"
    )

    # Glue job parameters
    parser.add_argument(
        "--bucket",
        type=str,
        default="prod-emea-lp-bi-datalake",
        help="S3 bucket for input/output data",
    )

    parser.add_argument(
        "--custom_date_filter",
        type=str,
        default="reporting_qdate = DATE '2025-01-01'",
        help="Custom date filter for SQL query (e.g., \"reporting_qdate = DATE '2025-01-01'\")",
    )

    parser.add_argument(
        "--base_prefix",
        type=str,
        default="data",
        help="Base S3 prefix for data",
    )

    parser.add_argument(
        "--out_prefix",
        type=str,
        default="data/ml-dataset",
        help="Output S3 prefix for feature-engineered data",
    )

    parser.add_argument(
        "--crawler_name",
        type=str,
        default="ML_DATASET",
        help="Glue crawler name",
    )

    # No custom filter parameter as per project requirements

    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Unique job ID for pipeline tracking and integration",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region",
    )

    parser.add_argument(
        "--run-glue-job",
        action="store_true",
        help="Run Glue job to generate data before processing",
    )

    # Data processing parameters
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Proportion of data to use for test set (default: 0.3)",
    )

    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=None,
        help="Optional S3 path or local directory containing the dataset (if not using Glue job output)",
    )

    # No initial sample size parameter as per project requirements

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="data/ml-dataset-stratified",
        help="S3 prefix for output directory containing train and test folders",
    )

    # Undersampling parameter
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=None,
        help="Ratio of negative to positive samples (if provided, undersampling is applied)",
    )

    args = parser.parse_args()

    # If requested, run Glue job to generate data
    if args.run_glue_job:
        job_args = {
            "--bucket": args.bucket,
            "--base_prefix": args.base_prefix,
            "--out_prefix": args.out_prefix,
            "--crawler_name": args.crawler_name,
            "--custom_date_filter": args.custom_date_filter,
        }

        # No custom filter handling - following the project convention

        logger.info(f"Using Glue job arguments: {job_args}")

        # Run Glue job with the parameters from command line
        try:
            job_id = run_glue_job(
                job_args=job_args,
                region=args.region,
                wait=True,
            )
            logger.info(f"Glue job completed (job ID: {job_id})")
        except Exception as e:
            logger.warning(
                f"Error running Glue job: {str(e)}. Will use default dataset."
            )

        # Construct the output path from the provided parameters
        bucket = args.bucket
        out_prefix = args.out_prefix
        output_path = f"s3://{bucket}/{out_prefix}"

        # In testing mode, we may not have actual output if job is still in WAITING state
        output_exists = check_glue_output(output_path, args.region)
        if output_exists:
            # Use Glue job output as input for data preparation
            input_data_dir = output_path
            logger.info(f"Using Glue job output as input: {input_data_dir}")
        else:
            # For testing purposes, use the default ML dataset location
            logger.info(
                f"No output found at {output_path}, using default dataset for testing"
            )
            input_data_dir = "s3://my-s3-bucket/data/ml-dataset-fe"
            logger.info(f"Using default dataset for testing: {input_data_dir}")

        # This line is now handled above in the if-else block
    else:
        # Use specified input directory or default to constructed path
        if args.input_data_dir:
            input_data_dir = args.input_data_dir
        else:
            input_data_dir = "s3://my-s3-bucket/data/ml-dataset"

        logger.info(f"Using specified input path: {input_data_dir}")

    # Load data
    df = load_data(input_data_dir, sample_size=None)

    # Use the global feature definitions from the top of the file
    label = LABEL
    num_features = NUM_FEATURES
    cat_features = CAT_FEATURES
    text_features = TEXT_FEATURES
    features = FEATURES

    # Prepare data
    logger.info("Preparing data for train-test split")
    y = df[label].copy()
    X = df[features].copy()

    # First, do train-test split without undersampling
    logger.info(f"Splitting data with test_size={args.test_size}, random_state=42")
    X_train_original, X_test, y_train_original, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Ensure correct dtypes for both splits
    X_train_original = X_train_original.copy()
    X_test = X_test.copy()
    X_train_original[num_features] = X_train_original[num_features].astype(float)
    X_test[num_features] = X_test[num_features].astype(float)

    for col in cat_features + text_features:
        X_train_original[col] = X_train_original[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Create original train dataframe with label
    original_train_df = X_train_original.copy()
    original_train_df[label] = y_train_original.copy()

    # Log statistics about original split
    logger.info(f"Original dataset size: {len(df)} rows")
    logger.info(f"Original training set: {len(original_train_df)} rows")
    logger.info(f"Test set: {len(X_test)} rows")
    orig_train_dist = y_train_original.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    logger.info(f"Original train class distribution: {orig_train_dist.to_dict()}")
    logger.info(f"Test class distribution: {test_dist.to_dict()}")

    # Initialize variables for if we don't do undersampling
    X_train = X_train_original.copy()
    y_train = y_train_original.copy()
    train_df = original_train_df.copy()

    # Apply undersampling only if sampling_ratio is provided
    apply_undersampling = args.sampling_ratio is not None
    if apply_undersampling:
        logger.info(f"Applying undersampling with ratio {args.sampling_ratio}")
        class_counts = y_train_original.value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")

        undersampler = RandomUnderSampler(
            sampling_strategy={
                0: int(class_counts[0] * args.sampling_ratio),
                1: class_counts[1],
            },
            random_state=42,
        )
        X_train, y_train = undersampler.fit_resample(X_train_original, y_train_original)

        # Create resampled train dataframe with label
        train_df = pd.DataFrame(X_train, copy=True)
        train_df[label] = y_train.copy()

        logger.info(f"Training set after undersampling: {X_train.shape}")
        logger.info(f"Resampled training set size: {len(train_df)} rows")
        resampled_train_dist = y_train.value_counts(normalize=True)
        logger.info(
            f"Resampled train class distribution: {resampled_train_dist.to_dict()}"
        )

    # Store feature metadata
    feature_metadata = {
        "num_features": num_features,
        "cat_features": cat_features,
        "text_features": text_features,
        "label": label,
    }

    # Create test dataframe with label
    test_df = X_test.copy()
    test_df[label] = y_test

    # Extract date from custom_date_filter (for fallback if job_id not provided)
    # Parse the custom_date_filter which is in format "= DATE 'YYYY-MM-DD'"
    if "DATE" in args.custom_date_filter:
        date_match = args.custom_date_filter.split("'")[1]
        date_str = date_match.replace("-", "")
    else:
        # Default to current date
        date_str = datetime.now().strftime("%Y%m%d")

    # Create a path component using just the job ID
    if args.job_id and args.job_id not in args.output_prefix:
        output_prefix = f"{args.output_prefix.rstrip('/')}/{args.job_id}"
    else:
        output_prefix = args.output_prefix.rstrip("/")

    # Prepare metadata with more comprehensive information
    metadata = {
        "creation_params": vars(args),
        "dataset_sizes": {
            "original_dataset_size": len(df),
            "original_train_size": len(original_train_df),
            "test_size": len(test_df),
            "resampled_train_size": len(train_df) if apply_undersampling else None,
        },
        "class_distributions": {
            "original_train_distribution": y_train_original.value_counts(
                normalize=True
            ).to_dict(),
            "resampled_train_distribution": y_train.value_counts(
                normalize=True
            ).to_dict()
            if apply_undersampling
            else None,
            "test_distribution": y_test.value_counts(normalize=True).to_dict(),
        },
        "feature_metadata": feature_metadata,
        "undersampling_applied": apply_undersampling,
        "sampling_ratio": args.sampling_ratio if apply_undersampling else None,
    }

    # Helper function to convert NumPy types to Python native types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {
                convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert any NumPy types and non-serializable objects
    metadata["creation_params"] = {
        k: str(v)
        if not isinstance(
            v, (str, int, float, bool, list, dict, np.integer, np.floating, np.bool_)
        )
        else convert_numpy_types(v)
        for k, v in metadata["creation_params"].items()
    }

    # Ensure all dictionary keys and values are JSON serializable
    metadata = convert_numpy_types(metadata)

    # Use the bucket from the command line args
    output_bucket = args.bucket

    # Save train and test datasets with metadata
    output_path = save_data(
        train_df, test_df, output_bucket, output_prefix, metadata, original_train_df
    )

    print(f"\n✅ Data preparation complete. Datasets saved to {output_path}")
    print(f"Train dataset: {len(train_df)} rows")
    print(f"Test dataset: {len(test_df)} rows")

    result = {
        "output_path": output_path,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "job_id": args.job_id or output_prefix.split("/")[-1],
    }
    print("\nPIPELINE_RESULT=" + json.dumps(result))

    # Return output path as last line for capture by SageMaker step
    print(output_path)


if __name__ == "__main__":
    main()

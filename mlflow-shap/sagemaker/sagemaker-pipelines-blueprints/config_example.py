"""
config_example.py

Example configuration file for SageMaker Pipeline.
Copy this file to config.py and customize for your use case.

Usage:
    from config import PipelineConfig
    config = PipelineConfig()
"""


class PipelineConfig:
    """
    Configuration class for SageMaker Pipeline components.
    Centralize all configuration in one place for easy management.
    """

    # ========================================================================
    # AWS Configuration
    # ========================================================================
    AWS_REGION = "us-east-1"
    AWS_ACCOUNT_ID = "123456789012"

    # IAM Role for SageMaker
    SAGEMAKER_ROLE = (
        f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole"
    )

    # S3 Configuration
    S3_BUCKET = "my-ml-bucket"
    S3_BASE_PREFIX = "ml-pipeline"
    S3_DATA_PREFIX = f"{S3_BASE_PREFIX}/data"
    S3_MODEL_PREFIX = f"{S3_BASE_PREFIX}/models"
    S3_OUTPUT_PREFIX = f"{S3_BASE_PREFIX}/outputs"

    # ========================================================================
    # MLflow Configuration
    # ========================================================================
    MLFLOW_TRACKING_URI = (
        f"arn:aws:sagemaker:{AWS_REGION}:{AWS_ACCOUNT_ID}:"
        "mlflow-tracking-server/my-mlflow-server"
    )
    MLFLOW_EXPERIMENT_NAME = "my-ml-experiment"
    MLFLOW_S3_BUCKET = f"sagemaker-{AWS_REGION}-{AWS_ACCOUNT_ID}"
    MLFLOW_EXPERIMENT_PATH = "mlflow-experiments/1"

    # ========================================================================
    # Network Configuration (VPC)
    # ========================================================================
    SECURITY_GROUP_IDS = [
        "sg-xxxxxxxxxxxxxxxxx",
    ]
    SUBNET_IDS = [
        "subnet-xxxxxxxxxxxxxxxxx",
        "subnet-yyyyyyyyyyyyyyyyy",
        "subnet-zzzzzzzzzzzzzzzzz",
    ]

    # ========================================================================
    # Docker Container Images
    # ========================================================================
    # Replace with your ECR repository URIs
    TRAINING_IMAGE_URI = (
        f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/"
        "ml-training-container:latest"
    )
    INFERENCE_IMAGE_URI = (
        f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/"
        "ml-inference-container:latest"
    )

    # ========================================================================
    # Feature Definitions
    # ========================================================================
    LABEL_COLUMN = "target"

    NUMERICAL_FEATURES = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    CATEGORICAL_FEATURES = [
        "category_1",
        "category_2",
    ]

    TEXT_FEATURES = [
        "text_feature_1",
    ]

    ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES

    # ========================================================================
    # Data Preparation Configuration
    # ========================================================================
    # Glue job for ETL (optional)
    GLUE_JOB_NAME = "my-etl-job"
    GLUE_CRAWLER_NAME = "my-data-crawler"
    GLUE_DATABASE_NAME = "my_database"

    # Train-test split
    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    # Class balancing
    APPLY_UNDERSAMPLING = True
    SAMPLING_RATIO = 0.1  # Ratio of majority to minority class

    # ========================================================================
    # Model Training Configuration
    # ========================================================================
    MODEL_TYPE = "catboost"  # Options: catboost, xgboost, lightgbm, sklearn

    # Default hyperparameters
    HYPERPARAMETERS = {
        "iterations": 500,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "early_stopping_rounds": 50,
    }

    # Training compute configuration
    TRAINING_INSTANCE_TYPE = "ml.m6i.32xlarge"
    TRAINING_INSTANCE_COUNT = 1
    TRAINING_VOLUME_SIZE_GB = 100

    # ========================================================================
    # Inference Configuration
    # ========================================================================
    # Compute configuration
    INFERENCE_INSTANCE_TYPE = "ml.m6i.8xlarge"
    INFERENCE_INSTANCE_COUNT = 4
    INFERENCE_VOLUME_SIZE_GB = 100

    # Processing configuration
    BATCH_SIZE = 10000
    NUM_CPUS_PER_INSTANCE = 32

    # Output configuration
    ENABLE_SHAP_VALUES = True
    ENABLE_GLUE_CRAWLER = True

    # ========================================================================
    # Redshift Configuration
    # ========================================================================
    REDSHIFT_HOST = "my-cluster.region.redshift.amazonaws.com"
    REDSHIFT_PORT = 5439
    REDSHIFT_DATABASE = "my_database"
    REDSHIFT_USER = "admin"
    REDSHIFT_PASSWORD_SSM_PATH = "/prod/redshift/password"

    # Target schema and table
    REDSHIFT_SCHEMA = "ml_predictions"
    REDSHIFT_TABLE = "model_scores"

    # IAM role for Redshift COPY
    REDSHIFT_IAM_ROLE = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/RedshiftCopyRole"

    # ========================================================================
    # Pipeline Configuration
    # ========================================================================
    ENABLE_CACHING = True
    CACHE_EXPIRE_AFTER = "1 day"

    # Pipeline parameters (can be overridden at execution time)
    PIPELINE_PARAMETERS = {
        "ModelVersion": "v1.0",
        "DateFilter": "date = DATE '2025-01-01'",
        "TestSize": TEST_SIZE,
        "SamplingRatio": SAMPLING_RATIO,
        "Iterations": HYPERPARAMETERS["iterations"],
        "LearningRate": HYPERPARAMETERS["learning_rate"],
        "Depth": HYPERPARAMETERS["depth"],
        "EarlyStopRounds": HYPERPARAMETERS["early_stopping_rounds"],
        "InstanceType": INFERENCE_INSTANCE_TYPE,
        "InstanceCount": INFERENCE_INSTANCE_COUNT,
        "BatchSize": BATCH_SIZE,
    }

    # ========================================================================
    # Monitoring and Logging
    # ========================================================================
    ENABLE_CLOUDWATCH_METRICS = True
    LOG_LEVEL = "INFO"

    # ========================================================================
    # Helper Methods
    # ========================================================================
    @classmethod
    def get_s3_path(cls, prefix: str, suffix: str = "") -> str:
        """
        Construct S3 path from components.

        Args:
            prefix: S3 prefix (e.g., 'data', 'models')
            suffix: Optional suffix to append

        Returns:
            Complete S3 path
        """
        path = f"s3://{cls.S3_BUCKET}/{prefix}"
        if suffix:
            path = f"{path}/{suffix}"
        return path

    @classmethod
    def get_training_data_path(cls, job_id: str) -> dict:
        """
        Get training data paths for a specific job.

        Args:
            job_id: Unique job identifier

        Returns:
            Dictionary with train and test paths
        """
        base_path = cls.get_s3_path(cls.S3_DATA_PREFIX, f"prepared/{job_id}")
        return {
            "train": f"{base_path}/train-resampled",
            "test": f"{base_path}/test",
        }

    @classmethod
    def get_model_output_path(cls) -> str:
        """Get model output path."""
        return cls.get_s3_path(cls.S3_MODEL_PREFIX, "artifacts")

    @classmethod
    def get_inference_output_path(cls, job_id: str) -> str:
        """Get inference output path for a specific job."""
        return cls.get_s3_path(cls.S3_OUTPUT_PREFIX, f"inference/{job_id}")

    @classmethod
    def validate_config(cls):
        """
        Validate configuration.
        Raises ValueError if configuration is invalid.
        """
        # Check required fields
        required_fields = [
            "AWS_REGION",
            "AWS_ACCOUNT_ID",
            "SAGEMAKER_ROLE",
            "S3_BUCKET",
        ]

        for field in required_fields:
            value = getattr(cls, field, None)
            if not value or value.startswith("YOUR_") or "123456789012" in value:
                raise ValueError(
                    f"Please configure {field} in config.py with your actual value"
                )

        # Validate instance types
        valid_instance_prefixes = ["ml."]
        if not cls.TRAINING_INSTANCE_TYPE.startswith(
            tuple[str, ...](valid_instance_prefixes)
        ):
            raise ValueError(
                f"Invalid training instance type: {cls.TRAINING_INSTANCE_TYPE}"
            )

        if not cls.INFERENCE_INSTANCE_TYPE.startswith(
            tuple[str, ...](valid_instance_prefixes)
        ):
            raise ValueError(
                f"Invalid inference instance type: {cls.INFERENCE_INSTANCE_TYPE}"
            )

        return True


# Example usage
if __name__ == "__main__":
    config = PipelineConfig()

    # Validate configuration
    try:
        config.validate_config()
        print("✅ Configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")

    # Print some config values
    print(f"\nS3 Bucket: {config.S3_BUCKET}")
    print(f"MLflow URI: {config.MLFLOW_TRACKING_URI}")
    print(f"Training Instance: {config.TRAINING_INSTANCE_TYPE}")
    print(f"Model Type: {config.MODEL_TYPE}")

    # Example: Get training data paths
    job_id = "20250101_120000"
    paths = config.get_training_data_path(job_id)
    print(f"\nTraining data paths for job {job_id}:")
    print(f"  Train: {paths['train']}")
    print(f"  Test: {paths['test']}")

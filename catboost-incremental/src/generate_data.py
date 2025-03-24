import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import os

app = typer.Typer()


def generate_synthetic_parquet(
    output_path: str,
    n_samples: int = 100_000,
    n_features: int = 10,
    chunk_size: int = 20_000,
    num_partitions: int = 10,
):
    """
    Generates synthetic classification data and saves it to a partitioned Parquet dataset using PyArrow.

    Args:
        output_path (str): Path to save the Parquet dataset.
        n_samples (int): Number of samples to generate.
        n_features (int): Number of feature columns.
        chunk_size (int): Number of rows per chunk.
        num_partitions (int): Number of partitions.
    """
    print(
        f"Generating synthetic data with {n_samples} samples, "
        f"{n_features} features, {num_partitions} partitions..."
    )

    feature_columns = [f"feat_{i}" for i in range(n_features)]
    all_columns = feature_columns + ["target", "partition_id"]

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    for i in range(0, n_samples, chunk_size):
        size = min(chunk_size, n_samples - i)  # Adjust last chunk size if needed
        X = np.random.rand(size, n_features)  # Feature matrix
        y = np.random.randint(0, 2, size)  # Binary target
        partition_id = np.random.randint(
            0, num_partitions, size
        )  # Assign random partition

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(
            pd.DataFrame(np.column_stack([X, y, partition_id]), columns=all_columns)
        )

        # Write to Parquet in partitions
        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=["partition_id"],
            existing_data_behavior="overwrite_or_ignore",
        )

        print(f"Saved {i + size}/{n_samples} samples...")

    print(f"Parquet dataset saved at: {output_path}")


@app.command()
def main(
    output_path: str = "data",
    n_samples: int = 100_000,
    n_features: int = 10,
    chunk_size: int = 20_000,
    num_partitions: int = 10,
):
    """
    CLI command to generate partitioned synthetic data and save it as Parquet.
    """
    generate_synthetic_parquet(
        output_path, n_samples, n_features, chunk_size, num_partitions
    )


if __name__ == "__main__":
    app()

import pyarrow.parquet as pq
from rich.progress import Progress

progress = Progress()


def read_partitioned_parquet(dataset_path, chunk_size=20_000):
    """
    Generator function to read data from a partitioned Parquet dataset in chunks.
    - dataset_path: Path to the partitioned dataset (e.g., "/data").
    - chunk_size: Number of rows per chunk.
    """
    dataset = pq.ParquetDataset(dataset_path)

    # for batch in tqdm(dataset.read().to_batches(chunk_size)):

    with progress:
        task = progress.add_task(
            "[cyan1]Reading Parquet...",
            total=len(dataset.read().to_batches(chunk_size)),
        )
        for batch in dataset.read().to_batches(chunk_size):
            progress.update(task, advance=1)
            df_chunk = batch.to_pandas()
            X = df_chunk.iloc[
                :, :-2
            ].values  # All columns except last (target) and partition_id
            y = df_chunk.iloc[:, -2].values  # Last column before partition_id as target
            yield X, y

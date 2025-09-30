import pyarrow.parquet as pq
from model_trainer import CatBoostTrainer
from data_loader import read_partitioned_parquet
from sklearn.model_selection import train_test_split
import time

# Path to dataset
dataset_path = "data"


def create_trainer():
    """Factory function to create a CatBoostTrainer instance."""
    return CatBoostTrainer(chunk_size=100_000)


def train():
    # sourcery skip: extract-duplicate-method, inline-immediately-returned-variable
    # 🚀 Load dataset metadata
    print("\n🚀 Loading dataset metadata...")
    dataset = pq.ParquetDataset(dataset_path)
    full_df = dataset.read().to_pandas()

    # 🚀 Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        full_df.drop(columns=["target", "partition_id"]),
        full_df["target"],
        test_size=0.2,
        random_state=42,
    )

    # 🚀 Standard Training (Not Incremental)
    print("\n🚀 Running Standard Training...")
    trainer = create_trainer()
    standard_time, standard_acc = trainer.train_normal(X_train, y_train, X_test, y_test)
    _evaluate_timing("Standard Training", standard_time, standard_acc)

    # 🚀 Standard Incremental Training
    print("\n🚀 Running Standard Incremental Training...")
    trainer = create_trainer()  # New instance for incremental training
    incremental_time, incremental_acc = trainer.train_incremental(
        read_partitioned_parquet(dataset_path, chunk_size=100_000), X_test, y_test
    )
    _evaluate_timing("Standard Incremental Training", incremental_time, incremental_acc)

    # 🚀 Optimized Incremental Training
    print("\n🚀 Running Optimized Incremental Training...")
    trainer = create_trainer()
    init_model = None
    start_time = time.time()

    for chunk_counter, (X_chunk, y_chunk) in enumerate(
        read_partitioned_parquet(dataset_path, chunk_size=100_000)
    ):
        if chunk_counter % 5 == 0:  # 🚀 Update model only every 5 chunks
            trainer.model.fit(X_chunk, y_chunk, init_model=init_model)
            init_model = trainer.model  # Store model after update

    optimized_time = time.time() - start_time
    optimized_acc = trainer.evaluate(X_test, y_test)
    _evaluate_timing("Optimized Incremental Training", optimized_time, optimized_acc)

    trainer.serialize("models/cb_model.cbm")


# An evaluation function to print the timing and accuracy
def _evaluate_timing(stage_name, elapsed_time, accuracy):
    """Prints training results without returning anything."""
    print(f"\n🚀 {stage_name} - Time: {elapsed_time:.2f}s, Accuracy: {accuracy:.4f}")
    # serialize the model


if __name__ == "__main__":
    train()

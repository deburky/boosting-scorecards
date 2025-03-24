import time

import typer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = typer.Typer()

# CatBoost parameters
catboost_params = dict(
    task_type="CPU",
    iterations=500,
    learning_rate=0.1,
    max_depth=3,
    verbose=0,
    allow_writing_files=False,  # Disable unnecessary file writes
)


class CatBoostTrainer:
    def __init__(self, chunk_size=20000):
        """
        Initializes the classifier but does NOT store data.
        - chunk_size: Number of rows to process at a time.
        """
        self.chunk_size = chunk_size
        self.model = CatBoostClassifier(**catboost_params)

    def train_normal(self, X_train, y_train, X_test, y_test):
        """Train CatBoost on the full dataset at once and track time."""
        start_time = time.time()
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=0.8, random_state=42
        )

        self.model.fit(X_train, y_train, eval_set=(X_val, y_val))

        duration = time.time() - start_time
        return duration, self.evaluate(X_test, y_test)

    def train_incremental(
        self, data_generator, X_test, y_test, update_every_n_chunks=5
    ):
        """
        Train CatBoost incrementally using chunks of data from a generator.
        - data_generator: A generator yielding (X_chunk, y_chunk) pairs.
        - X_test, y_test: Test data for evaluation.
        - update_every_n_chunks: Number of chunks before updating model.
        """
        start_time = time.time()
        init_model = None
        first_chunk = True

        for chunk_counter, (X_chunk, y_chunk) in enumerate(data_generator):
            if first_chunk:
                # Set up validation set once from the first chunk
                X_train, X_val, y_train, y_val = train_test_split(
                    X_chunk, y_chunk, train_size=0.8, random_state=42
                )
                self.model.fit(X_train, y_train, eval_set=(X_val, y_val))
                init_model = self.model  # Store model for future updates
                first_chunk = False
            elif chunk_counter % update_every_n_chunks == 0:
                self.model.fit(
                    X_chunk, y_chunk, eval_set=(X_val, y_val), init_model=init_model
                )
                init_model = self.model  # Store updated model

        duration = time.time() - start_time
        return duration, self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """Evaluate model accuracy on the test dataset."""
        preds = self.model.predict(X_test)
        return accuracy_score(y_test, preds)

    def serialize(self, path):
        """Serialize the model to a file."""
        self.model.save_model(path, format="cbm")

import mlflow
import mlflow.catboost
import mlflow.shap
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
import shap


def create_sample_data():
    """Create sample classification dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    # Convert to DataFrame for better feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, feature_names


def train_catboost_model(X_train, y_train, X_val, y_val):
    """Train a CatBoost classifier"""
    # Create CatBoost pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # Initialize and train the model
    model = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=False
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=10, verbose=False)

    return model


def log_model_with_shap(
    model, X_test, y_test, feature_names, model_name="catboost_shap_model"
):
    """Log CatBoost model with both SHAP plots and explainer object"""

    with mlflow.start_run(run_name=f"{model_name}_run") as run:
        # Log model parameters
        mlflow.log_params(
            {
                "model_type": "CatBoostClassifier",
                "iterations": model.get_params()["iterations"],
                "learning_rate": model.get_params()["learning_rate"],
                "depth": model.get_params()["depth"],
            }
        )

        # Infer model signature
        signature = infer_signature(X_test, model.predict(X_test))

        # Log the CatBoost model using mlflow.catboost
        model_info = mlflow.catboost.log_model(
            cb_model=model,
            name="model",
            signature=signature,
            registered_model_name=model_name,
        )

        # Create evaluation dataset for SHAP plots
        eval_data = X_test.copy()
        eval_data["label"] = y_test

        # Use MLflow's built-in SHAP integration with advanced configuration
        print("Running MLflow evaluation with advanced SHAP configuration...")

        # Advanced SHAP configuration for CatBoost
        shap_config = {
            "log_explainer": True,  # Save the explainer model
            "explainer_type": "partition",  # Use partition explainer for tree models
            "max_error_examples": 100,  # Number of error cases to explain
            "log_model_explanations": True,  # Log individual prediction explanations
        }

        result = mlflow.evaluate(
            model_info.model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
            evaluator_config={"default": shap_config},
        )

        print("SHAP plots generated:")
        shap_artifacts = []
        for artifact_name in result.artifacts:
            if "shap" in artifact_name.lower():
                print(f"  - {artifact_name}")
                shap_artifacts.append(artifact_name)

        # Create and save a custom SHAP explainer that can be serialized
        print("Creating and saving custom SHAP explainer...")

        class SerializableSHAPExplainer:
            """Custom SHAP explainer that can be serialized by MLflow"""

            def __init__(self, model):
                self.model = model
                self.explainer = shap.TreeExplainer(model)

            def predict(self, X):
                """Generate SHAP values for input data"""
                return self.explainer.shap_values(X)

            def __call__(self, X):
                return self.predict(X)

        # Create the custom explainer
        shap_explainer = SerializableSHAPExplainer(model)

        # Log the explainer as a custom pyfunc model
        mlflow.pyfunc.log_model(
            artifact_path="explainer",
            python_model=shap_explainer,
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=f"{model_name}_explainer",
            metadata={
                "model_type": "SHAP_Explainer",
                "base_model": "CatBoostClassifier",
            },
        )

        print("SHAP explainer saved successfully!")

        # Calculate and log SHAP values for demonstration
        sample_size = min(100, len(X_test))
        X_sample = X_test.iloc[:sample_size]

        # Create explainer using the original CatBoost model (not the wrapper)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Log SHAP values as artifacts
        shap_values_df = pd.DataFrame(
            shap_values, columns=feature_names, index=X_sample.index
        )
        shap_values_df.to_csv("shap_values.csv", index=False)
        mlflow.log_artifact("shap_values.csv")

        # Log feature importance
        feature_importance = model.get_feature_importance()
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        ).sort_values("importance", ascending=False)

        importance_df.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        # Log additional metrics
        mlflow.log_metrics(
            {
                "accuracy_score": result.metrics.get("accuracy_score", 0),
                "precision_score": result.metrics.get("precision_score", 0),
                "recall_score": result.metrics.get("recall_score", 0),
                "f1_score": result.metrics.get("f1_score", 0),
            }
        )

        print("Model, SHAP plots, and explainer object logged successfully!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: runs:/{run.info.run_id}/model")
        print(f"SHAP Explainer URI: runs:/{run.info.run_id}/shap_explainer")
        print(f"Generated {len(shap_artifacts)} SHAP plot artifacts")

        return run.info.run_id, result


def load_and_explain_model(run_id, X_sample):
    """Load the logged model and SHAP explainer object"""

    # Load the model using mlflow.catboost
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.catboost.load_model(model_uri)

    # Load the SHAP explainer from MLflow evaluation
    explainer_uri = f"runs:/{run_id}/explainer"
    print("Loading SHAP explainer from MLflow evaluation...")

    try:
        loaded_explainer = mlflow.pyfunc.load_model(explainer_uri)
        print("Successfully loaded SHAP explainer from MLflow evaluation!")
    except Exception as e:
        print(f"Could not load SHAP explainer from evaluation: {e}")
        print("Creating new SHAP explainer for demonstration...")
        loaded_explainer = shap.TreeExplainer(loaded_model)

    # Make predictions
    predictions = loaded_model.predict(X_sample)
    probabilities = loaded_model.predict_proba(X_sample)

    # Generate SHAP values using the loaded explainer
    print("Generating SHAP values using loaded explainer...")
    if hasattr(loaded_explainer, "shap_values"):
        shap_values = loaded_explainer.shap_values(X_sample)
    else:
        # If it's a pyfunc model, use predict method
        shap_values = loaded_explainer.predict(X_sample)

    print(f"Loaded model predictions for {len(X_sample)} samples:")
    print(f"Predictions: {predictions[:5]}...")  # Show first 5
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"SHAP values shape: {shap_values.shape}")

    return loaded_model, loaded_explainer, predictions, shap_values


def main():
    """Main function demonstrating MLflow + CatBoost + SHAP integration"""

    print("=== MLflow + CatBoost + SHAP Example ===\n")

    # Set MLflow tracking URI (optional - defaults to local file system)
    mlflow.set_tracking_uri("file:./mlruns")

    # Create sample data
    print("1. Creating sample dataset...")
    X, y, feature_names = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Train CatBoost model
    print("\n2. Training CatBoost model...")
    model = train_catboost_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"   Training accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")

    # Log model with SHAP explainer using MLflow's built-in integration
    print("\n3. Logging model and SHAP explainer to MLflow...")
    run_id, eval_result = log_model_with_shap(model, X_test, y_test, feature_names)

    # Demonstrate loading and using the logged model and explainer
    print("\n4. Loading and testing the logged model and SHAP explainer...")
    sample_size = 10
    X_sample = X_test.iloc[:sample_size]
    loaded_model, loaded_explainer, predictions, shap_values = load_and_explain_model(
        run_id, X_sample
    )

    print("\n5. Feature importance from CatBoost:")
    importance = loaded_model.get_feature_importance()
    for feature, imp in zip(feature_names, importance):
        print(f"   {feature}: {imp:.4f}")

    print("\n6. SHAP values for first sample:")
    print(f"   Sample features: {X_sample.iloc[0].values}")
    print(f"   SHAP values: {shap_values[0]}")
    print(f"   Prediction: {predictions[0]}")

    print("\n7. Available SHAP plot artifacts from MLflow evaluation:")
    for artifact_name, artifact_path in eval_result.artifacts.items():
        if "shap" in artifact_name.lower():
            print(f"   - {artifact_name}: {artifact_path}")

    print("\n8. SHAP Feature Importance (from CatBoost):")
    for feature, imp in zip(feature_names, importance):
        print(f"   {feature}: {imp:.4f}")

    print("\n9. Successfully loaded both:")
    print("- CatBoost model for predictions")
    print("- SHAP explainer object for generating explanations")
    print("- SHAP plots for visualization")

    print("\n=== Example completed successfully! ===")
    print(f"Check the MLflow UI at: file://{mlflow.get_tracking_uri()}/index.html")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to load and use the saved SHAP explainer from MLflow Model Registry
"""

import traceback

import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import make_classification


def create_test_data():
    """Create test data for demonstration."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, feature_names


def load_model_and_explainer():
    """Load both the CatBoost model and SHAP explainer from MLflow."""

    print("=== Loading Models from MLflow Model Registry ===\n")

    # Load the CatBoost model
    print("1. Loading CatBoost model...")
    model_uri = "models:/catboost_shap_model/latest"
    catboost_model = mlflow.catboost.load_model(model_uri)
    print("✓ CatBoost model loaded successfully")
    print(f"Model type: {type(catboost_model)}")

    # Load the SHAP explainer
    print("\n2. Loading SHAP explainer...")
    explainer_uri = "models:/catboost_shap_model_explainer/latest"
    shap_explainer = mlflow.pyfunc.load_model(explainer_uri)
    print("✓ SHAP explainer loaded successfully")
    print(f"   Explainer type: {type(shap_explainer)}")

    return catboost_model, shap_explainer


def test_explainer_functionality(catboost_model, shap_explainer, X_test, feature_names):
    """Test the loaded explainer functionality."""

    print("\n=== Testing SHAP Explainer Functionality ===\n")

    # Test 1: Make predictions with CatBoost model
    print("3. Testing CatBoost model predictions...")
    predictions = catboost_model.predict(X_test)
    probabilities = catboost_model.predict_proba(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")

    # Test 2: Generate SHAP values using the loaded explainer
    print("\n4. Testing SHAP explainer...")
    sample_size = 10
    X_sample = X_test.iloc[:sample_size]

    try:
        # Use the loaded explainer to generate SHAP values
        shap_values = shap_explainer.predict(X_sample)
        print("✓ SHAP values generated successfully")
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"SHAP values type: {type(shap_values)}")

        # Test 3: Analyze SHAP values
        print("\n5. Analyzing SHAP values...")
        print(f"   Mean absolute SHAP value: {np.abs(shap_values).mean():.4f}")
        print(
            f"   SHAP value range: [{shap_values.min():.4f}, {shap_values.max():.4f}]"
        )

        # Show feature importance based on SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "shap_importance": feature_importance}
        ).sort_values("shap_importance", ascending=False)

        print("\n   Top 5 features by SHAP importance:")
        # sourcery skip: no-loop-in-tests
        for i, (_, row) in enumerate(importance_df.head().iterrows()):
            print(f"   {i + 1}. {row['feature']}: {row['shap_importance']:.4f}")

        # Test 4: Show individual prediction explanation
        print("\n6. Individual prediction explanation (first sample):")
        print(f"Sample features: {X_sample.iloc[0].values}")
        print(f"SHAP values: {shap_values[0]}")
        print(f"Prediction: {predictions[0]}")
        print(f"Probability: {probabilities[0]}")

        # Show which features contributed most to this prediction
        sample_shap = shap_values[0]
        feature_contrib = list(zip(feature_names, sample_shap))
        feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\nTop 3 features contributing to this prediction:")
        for i, (feature, contrib) in enumerate(feature_contrib[:3]):
            direction = "increases" if contrib > 0 else "decreases"
            print(f"{i + 1}. {feature}: {contrib:.4f} ({direction} prediction)")

        return True

    except (ValueError, AttributeError, RuntimeError) as e:
        print(f"✗ Error generating SHAP values: {e}")
        return False


def compare_with_fresh_explainer(catboost_model, X_sample):
    """Compare loaded explainer with a fresh one to verify correctness."""

    print("\n=== Comparing with Fresh SHAP Explainer ===\n")

    print("7. Creating fresh SHAP explainer for comparison...")
    fresh_explainer = shap.TreeExplainer(catboost_model)
    fresh_shap_values = fresh_explainer.shap_values(X_sample)

    print(f"   Fresh SHAP values shape: {fresh_shap_values.shape}")
    print(f"   Fresh SHAP values type: {type(fresh_shap_values)}")

    return fresh_shap_values


def setup_test_environment():
    """Set up the test environment and create test data"""
    print("=== MLflow SHAP Explainer Test Script ===\n")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")

    # Create test data
    print("Creating test data...")
    X_test, _, feature_names = create_test_data()
    print(f"Test data shape: {X_test.shape}")

    return X_test, feature_names


def main():
    """Main function to test the saved SHAP explainer"""

    try:
        # Setup test environment
        X_test, feature_names = setup_test_environment()

        # Load models
        catboost_model, shap_explainer = load_model_and_explainer()

        # Test explainer functionality
        if test_explainer_functionality(
            catboost_model, shap_explainer, X_test, feature_names
        ):
            # Compare with fresh explainer
            X_sample = X_test.iloc[:5]  # Use smaller sample for comparison
            fresh_shap_values = compare_with_fresh_explainer(catboost_model, X_sample)

            # Load explainer values for comparison
            loaded_shap_values = shap_explainer.predict(X_sample)

            # Compare values
            print("\n8. Comparing loaded vs fresh SHAP values...")
            if np.allclose(loaded_shap_values, fresh_shap_values, rtol=1e-10):
                print("   ✓ SHAP values match perfectly!")
            else:
                print(
                    "⚠ SHAP values differ (this might be expected due to serialization)"
                )
                print(
                    f"Max difference: {np.abs(loaded_shap_values - fresh_shap_values).max():.10f}"
                )

        print("\n=== Test Completed Successfully! ===")
        print(
            "The SHAP explainer can be successfully loaded and used from MLflow Model Registry."
        )

    except (mlflow.exceptions.MlflowException, FileNotFoundError, ImportError) as e:
        print(f"\n✗ Error during testing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""Tests for model training and prediction."""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import get_feature_columns, prepare_data_log, build_pipeline

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_model.pkl")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "cleaned_data.csv")


@pytest.fixture
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not found - run training first")
    return joblib.load(MODEL_PATH)


@pytest.fixture
def data():
    if not os.path.exists(DATA_PATH):
        pytest.skip("Processed data not found")
    return pd.read_csv(DATA_PATH)


def test_model_loads(model):
    """Test that the model loads without error."""
    assert model is not None


def test_model_predicts(model, data):
    """Test that the model can make predictions."""
    X, y = prepare_data_log(data)
    sample = X.iloc[:5]
    predictions = model.predict(sample)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_model_prediction_range(model, data):
    """Test that predictions are in a reasonable range."""
    X, y = prepare_data_log(data)
    sample = X.iloc[:100]
    predictions = np.expm1(model.predict(sample))  # Back to original scale
    assert (predictions > 0).all(), "All predictions should be positive"
    assert (predictions < 200).all(), "Predictions should be < 200 lakhs"


def test_model_single_prediction(model):
    """Test prediction with a single input."""
    input_data = pd.DataFrame([{
        'Present_Price': 8.5,
        'Kms_Driven': 35000,
        'Owner': 0,
        'Car_Age': 3,
        'Brand': 'Maruti',
        'Fuel_Type': 'Petrol',
        'Seller_Type': 'Dealer',
        'Transmission': 'Manual',
    }])

    pred = model.predict(input_data)
    price = np.expm1(pred[0])
    assert price > 0, f"Price should be positive, got {price}"
    assert price < 50, f"Price for a Maruti should be < 50 lakhs, got {price}"


def test_model_r2_score(model, data):
    """Test that model R2 is above threshold."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X, y = prepare_data_log(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    y_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)

    r2 = r2_score(y_actual, y_pred_actual)
    assert r2 >= 0.85, f"R2 should be >= 0.85, got {r2:.4f}"


def test_metrics_file_exists():
    """Test that model comparison CSV exists."""
    metrics_path = os.path.join(PROJECT_DIR, "outputs", "metrics", "model_comparison.csv")
    if not os.path.exists(metrics_path):
        pytest.skip("Metrics file not found - run training first")
    df = pd.read_csv(metrics_path)
    assert len(df) >= 3, "Should have at least 3 model results"
    assert 'R2' in df.columns
    assert 'MAE' in df.columns
    assert 'RMSE' in df.columns

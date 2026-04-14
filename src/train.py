"""
Model training module.
Trains Linear Regression, Random Forest, and XGBoost.
Compares models and saves the best one.
"""
import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow `python3 src/train.py` from the repository root.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.feature_engineering import build_pipeline, prepare_data_log, TARGET


def train_and_evaluate(df, models_dir, metrics_dir):
    """Train all models, compare, and save the best."""

    X, y = prepare_data_log(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    print()

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0
        ),
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = build_pipeline(model)

        # Time the training
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = round(time.time() - start, 2)

        # Predictions (convert back from log scale)
        y_pred_log = pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_test)

        # Metrics
        mae = round(mean_absolute_error(y_actual, y_pred), 4)
        rmse = round(np.sqrt(mean_squared_error(y_actual, y_pred)), 4)
        r2 = round(r2_score(y_actual, y_pred), 4)

        # Cross-validation (on log scale)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
                                     scoring='r2', n_jobs=-1)
        cv_mean = round(cv_scores.mean(), 4)
        cv_std = round(cv_scores.std(), 4)

        result = {
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2_Mean': cv_mean,
            'CV_R2_Std': cv_std,
            'Train_Time_s': train_time,
        }
        results.append(result)

        print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | "
              f"CV R2: {cv_mean:.4f} (+/- {cv_std:.4f}) | Time: {train_time}s")

        # Save each pipeline
        model_path = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(pipeline, model_path)

    # Comparison table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Save metrics
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "model_comparison.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    # Find best model by R2
    best_idx = results_df['R2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_r2 = results_df.loc[best_idx, 'R2']

    print(f"\nBest model: {best_model_name} (R2 = {best_r2})")

    # Copy best model as best_model.pkl
    best_source = os.path.join(models_dir, f"{best_model_name.lower().replace(' ', '_')}.pkl")
    best_dest = os.path.join(models_dir, "best_model.pkl")
    best_pipeline = joblib.load(best_source)
    joblib.dump(best_pipeline, best_dest)
    print(f"Best model saved to {best_dest}")

    return results_df, best_pipeline, X_test, y_test


if __name__ == "__main__":
    processed_path = os.path.join(PROJECT_DIR, "data", "processed", "cleaned_data.csv")
    models_dir = os.path.join(PROJECT_DIR, "models")
    metrics_dir = os.path.join(PROJECT_DIR, "outputs", "metrics")

    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(processed_path)
    train_and_evaluate(df, models_dir, metrics_dir)

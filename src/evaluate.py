"""
Evaluation module.
Generates all visualizations and evaluation plots.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow `python3 src/evaluate.py` from the repository root.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.feature_engineering import prepare_data_log, get_feature_columns, NUMERIC_FEATURES


def generate_eda_plots(df, figures_dir):
    """Generate all EDA visualizations."""
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # 1. Price distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['Selling_Price'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of Selling Price', fontsize=14)
    axes[0].set_xlabel('Selling Price (Lakhs)')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(np.log1p(df['Selling_Price']), bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribution of Log(Selling Price)', fontsize=14)
    axes[1].set_xlabel('Log(Selling Price)')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '01_price_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [1/8] Price distribution plot saved")

    # 2. Correlation heatmap
    numeric_df = df[['Selling_Price', 'Present_Price', 'Kms_Driven', 'Owner', 'Car_Age']]
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [2/8] Correlation heatmap saved")

    # 3. Price by fuel type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Fuel_Type', y='Selling_Price', palette='Set2', ax=ax)
    ax.set_title('Selling Price by Fuel Type', fontsize=14)
    ax.set_xlabel('Fuel Type')
    ax.set_ylabel('Selling Price (Lakhs)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '03_price_by_fuel.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [3/8] Price by fuel type saved")

    # 4. Price by transmission
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Transmission', y='Selling_Price', palette='Set3', ax=ax)
    ax.set_title('Selling Price by Transmission', fontsize=14)
    ax.set_xlabel('Transmission')
    ax.set_ylabel('Selling Price (Lakhs)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_price_by_transmission.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4/8] Price by transmission saved")

    # 5. Price vs Car Age scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Car_Age'], df['Selling_Price'], alpha=0.3, s=10, color='steelblue')
    ax.set_title('Selling Price vs Car Age', fontsize=14)
    ax.set_xlabel('Car Age (Years)')
    ax.set_ylabel('Selling Price (Lakhs)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '05_price_vs_age.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [5/8] Price vs age scatter saved")

    # 6. Price vs Kms Driven
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Kms_Driven'], df['Selling_Price'], alpha=0.3, s=10, color='coral')
    ax.set_title('Selling Price vs Kilometers Driven', fontsize=14)
    ax.set_xlabel('Kilometers Driven')
    ax.set_ylabel('Selling Price (Lakhs)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '06_price_vs_kms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [6/8] Price vs kms scatter saved")

    # 7. Top brands by count
    fig, ax = plt.subplots(figsize=(12, 6))
    brand_counts = df['Brand'].value_counts()
    brand_counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.7, ax=ax)
    ax.set_title('Number of Cars by Brand', fontsize=14)
    ax.set_xlabel('Brand')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '07_brand_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [7/8] Brand distribution saved")

    # 8. Price by seller type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Seller_Type', y='Selling_Price', palette='pastel', ax=ax)
    ax.set_title('Selling Price by Seller Type', fontsize=14)
    ax.set_xlabel('Seller Type')
    ax.set_ylabel('Selling Price (Lakhs)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '08_price_by_seller.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [8/8] Price by seller type saved")

    print(f"\nAll EDA plots saved to {figures_dir}")


def generate_evaluation_plots(model_path, df, figures_dir):
    """Generate model evaluation plots."""
    os.makedirs(figures_dir, exist_ok=True)

    pipeline = joblib.load(model_path)
    X, y = prepare_data_log(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predictions
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    # 9. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_actual, y_pred, alpha=0.4, s=15, color='steelblue', label='Predictions')
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_title('Actual vs Predicted Selling Price', fontsize=14)
    ax.set_xlabel('Actual Price (Lakhs)')
    ax.set_ylabel('Predicted Price (Lakhs)')
    ax.legend()

    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    ax.text(0.05, 0.90, f'R² = {r2:.4f}\nMAE = {mae:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '09_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [9] Actual vs predicted plot saved")

    # 10. Residual distribution
    residuals = y_actual - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Residuals vs Predicted', fontsize=14)
    axes[0].set_xlabel('Predicted Price (Lakhs)')
    axes[0].set_ylabel('Residual')

    axes[1].hist(residuals, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_title('Residual Distribution', fontsize=14)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '10_residuals.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [10] Residual plots saved")

    # 11. Feature importance (from the model inside the pipeline)
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features

    fig, ax = plt.subplots(figsize=(12, 8))
    names = [feature_names[i] for i in indices]
    # Clean up names
    names = [n.replace('num__', '').replace('cat__', '') for n in names]
    values = [importances[i] for i in indices]

    bars = ax.barh(range(len(names)), values, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title('Top 20 Feature Importances (Best Model)', fontsize=14)
    ax.set_xlabel('Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '11_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [11] Feature importance plot saved")

    # 12. Learning curves
    fig, ax = plt.subplots(figsize=(10, 6))
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='steelblue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='coral')
    ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training Score')
    ax.plot(train_sizes, val_mean, 'o-', color='coral', label='Validation Score')
    ax.set_title('Learning Curves (Best Model)', fontsize=14)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R² Score')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '12_learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [12] Learning curves saved")

    print(f"\nAll evaluation plots saved to {figures_dir}")

    # Print final metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    print(f"\n{'='*50}")
    print(f"FINAL MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  {mae:.4f} lakhs")
    print(f"RMSE: {rmse:.4f} lakhs")
    print(f"{'='*50}")

    return r2, mae, rmse


if __name__ == "__main__":
    processed_path = os.path.join(PROJECT_DIR, "data", "processed", "cleaned_data.csv")
    model_path = os.path.join(PROJECT_DIR, "models", "best_model.pkl")
    figures_dir = os.path.join(PROJECT_DIR, "outputs", "figures")

    df = pd.read_csv(processed_path)

    print("Generating EDA plots...")
    generate_eda_plots(df, figures_dir)

    print("\nGenerating evaluation plots...")
    generate_evaluation_plots(model_path, df, figures_dir)

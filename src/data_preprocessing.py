"""
Data preprocessing module for car price prediction.
Handles cleaning, null handling, outlier removal, and feature extraction.
"""
import pandas as pd
import numpy as np
import os


def load_raw_data(filepath):
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def extract_brand(df):
    """Extract car brand from Car_Name."""
    df = df.copy()
    df['Brand'] = df['Car_Name'].apply(lambda x: x.split(' ')[0])
    # Standardize brand names
    brand_map = {
        'Maruti': 'Maruti', 'Hyundai': 'Hyundai', 'Honda': 'Honda',
        'Toyota': 'Toyota', 'Ford': 'Ford', 'Volkswagen': 'Volkswagen',
        'Tata': 'Tata', 'Mahindra': 'Mahindra', 'Chevrolet': 'Chevrolet',
        'Renault': 'Renault', 'BMW': 'BMW', 'Audi': 'Audi',
        'Mercedes-Benz': 'Mercedes-Benz', 'Skoda': 'Skoda', 'Nissan': 'Nissan',
    }
    df['Brand'] = df['Brand'].map(brand_map).fillna('Other')
    return df


def add_car_age(df, current_year=2026):
    """Add car_age feature."""
    df = df.copy()
    df['Car_Age'] = current_year - df['Year']
    return df


def handle_missing_values(df):
    """Handle missing values - median for numeric, mode for categorical."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before > after:
        print(f"Removed {before - after} duplicate rows")
    return df


def remove_outliers(df):
    """Remove outliers from price and kms_driven."""
    df = df.copy()
    before = len(df)

    # Remove zero or negative prices
    df = df[df['Selling_Price'] > 0]

    # Remove unrealistic kms (negative or > 1M)
    df = df[df['Kms_Driven'] > 0]
    df = df[(df['Kms_Driven'] < 1000000)]

    # Remove extreme price outliers (beyond 3 IQR)
    Q1 = df['Selling_Price'].quantile(0.01)
    Q3 = df['Selling_Price'].quantile(0.99)
    df = df[(df['Selling_Price'] >= Q1) & (df['Selling_Price'] <= Q3)]

    after = len(df)
    print(f"Outlier removal: {before} -> {after} rows ({before - after} removed)")
    return df


def preprocess_data(raw_path, processed_path):
    """Full preprocessing pipeline."""
    df = load_raw_data(raw_path)
    df = extract_brand(df)
    df = add_car_age(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)

    # Drop Car_Name (we extracted Brand) and Year (we have Car_Age)
    df = df.drop(columns=['Car_Name', 'Year'])

    # Reorder columns
    cols = ['Brand', 'Present_Price', 'Kms_Driven', 'Fuel_Type',
            'Seller_Type', 'Transmission', 'Owner', 'Car_Age', 'Selling_Price']
    df = df[cols]

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"\nProcessed data saved: {processed_path}")
    print(f"Final shape: {df.shape}")
    print(f"Null values: {df.isnull().sum().sum()}")
    print(f"Dtypes:\n{df.dtypes}")
    return df


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = os.path.join(project_dir, "data", "raw", "car_data.csv")
    processed = os.path.join(project_dir, "data", "processed", "cleaned_data.csv")
    preprocess_data(raw, processed)

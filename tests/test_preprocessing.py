"""Tests for data preprocessing module."""
import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import (
    extract_brand, handle_missing_values, remove_duplicates,
    remove_outliers, add_car_age, preprocess_data
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Car_Name': ['Maruti Swift', 'Hyundai i20', 'Honda City', 'Toyota Fortuner'],
        'Year': [2020, 2018, 2015, 2022],
        'Selling_Price': [5.5, 7.2, 4.8, 30.0],
        'Present_Price': [7.0, 9.5, 8.0, 35.0],
        'Kms_Driven': [15000, 35000, 60000, 5000],
        'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Diesel'],
        'Seller_Type': ['Dealer', 'Individual', 'Dealer', 'Dealer'],
        'Transmission': ['Manual', 'Manual', 'Automatic', 'Automatic'],
        'Owner': [0, 1, 0, 0],
    })


def test_extract_brand(sample_data):
    df = extract_brand(sample_data)
    assert 'Brand' in df.columns
    assert list(df['Brand']) == ['Maruti', 'Hyundai', 'Honda', 'Toyota']


def test_handle_missing_values():
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': ['x', None, 'y', 'z'],
    })
    result = handle_missing_values(df)
    assert result.isnull().sum().sum() == 0


def test_remove_duplicates():
    df = pd.DataFrame({'A': [1, 1, 2], 'B': ['x', 'x', 'y']})
    result = remove_duplicates(df)
    assert len(result) == 2


def test_remove_outliers():
    df = pd.DataFrame({
        'Selling_Price': [5, 10, 15, 0, -5, 100],
        'Kms_Driven': [10000, 20000, 30000, 10000, 10000, 10000],
    })
    result = remove_outliers(df)
    assert (result['Selling_Price'] > 0).all()
    assert (result['Kms_Driven'] > 0).all()


def test_add_car_age(sample_data):
    df = add_car_age(sample_data, current_year=2026)
    assert 'Car_Age' in df.columns
    assert df.loc[0, 'Car_Age'] == 6  # 2026 - 2020
    assert df.loc[3, 'Car_Age'] == 4  # 2026 - 2022


def test_full_preprocessing():
    """Test the full preprocessing pipeline on real data."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_dir, "data", "raw", "car_data.csv")
    processed_path = os.path.join(project_dir, "data", "processed", "test_cleaned.csv")

    if not os.path.exists(raw_path):
        pytest.skip("Raw data not found")

    df = preprocess_data(raw_path, processed_path)

    # Exit criteria checks
    assert df.isnull().sum().sum() == 0, "There should be no null values"
    assert len(df) > 7000, f"Should have > 7000 rows, got {len(df)}"
    assert 'Brand' in df.columns, "Brand column should exist"
    assert 'Car_Age' in df.columns, "Car_Age column should exist"
    assert 'Car_Name' not in df.columns, "Car_Name should be dropped"
    assert (df['Selling_Price'] > 0).all(), "All prices should be positive"

    # Cleanup
    if os.path.exists(processed_path):
        os.remove(processed_path)

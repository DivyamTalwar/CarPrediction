"""
Feature engineering module.
Builds sklearn Pipeline with ColumnTransformer for preprocessing + model.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Define feature columns
NUMERIC_FEATURES = ['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age']
CATEGORICAL_FEATURES = ['Brand', 'Fuel_Type', 'Seller_Type', 'Transmission']
TARGET = 'Selling_Price'


def get_feature_columns():
    """Return feature column names."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    """Build a ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder='drop'
    )
    return preprocessor


def build_pipeline(model):
    """Build a full pipeline: preprocessor + model."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


def prepare_data(df):
    """Split into features and target."""
    X = df[get_feature_columns()]
    y = df[TARGET]
    return X, y


def prepare_data_log(df):
    """Split into features and log-transformed target."""
    X = df[get_feature_columns()]
    y = np.log1p(df[TARGET])  # log(1 + price) to handle small values
    return X, y

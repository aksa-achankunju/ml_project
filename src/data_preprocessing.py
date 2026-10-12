"""
RoadSafe AI - Data Preprocessing & Feature Pipeline
Addresses CO1 (Feature Pipeline) and CO6 (Training-Serving Skew Prevention).
Provides unified ColumnTransformer pipeline, dataset splitting, and feature store interface.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Feature Specifications
NUMERICAL_FEATURES = [
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Hour',
    'Month',
    'Start_Lat',
    'Start_Lng'
]

CATEGORICAL_FEATURES = [
    'Weather_Condition',
    'Sunrise_Sunset',
    'State_Clean'
]

BINARY_FEATURES = [
    'Crossing',
    'Give_Way',
    'Junction',
    'No_Exit',
    'Railway',
    'Roundabout',
    'Station',
    'Stop',
    'Traffic_Calming',
    'Traffic_Signal',
    'Amenity',
    'Bump',
    'Is_Weekend',
    'Rush_Hour'
]

ALL_FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

def create_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer ensuring training-serving skew prevention.
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder (handle_unknown='ignore')
    - Binary: Passthrough
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def load_data(filepath=None):
    """
    Loads the cleaned sample dataset.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'roadsafe_clean_sample.csv')
    df = pd.read_csv(filepath)
    return df

def get_train_val_test_data(df=None, test_size=0.15, val_size=0.15, random_state=42):
    """
    Performs stratified train/val/test splits (CO5 requirement).
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, y_reg_train, y_reg_val, y_reg_test)
    where y is Severity (multiclass 1-4) and y_reg is Duration_Minutes.
    """
    if df is None:
        df = load_data()

    X = df[ALL_FEATURE_COLS].copy()
    y_clf = df['Severity'].values
    y_reg = df['Duration_Minutes'].values

    # Step 1: Split into Train + Val (85%) and Test (15%)
    X_temp, X_test, y_clf_temp, y_clf_test, y_reg_temp, y_reg_test = train_test_split(
        X, y_clf, y_reg,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf
    )

    # Step 2: Split Temp into Train (70%) and Val (15% of total)
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_clf_train, y_clf_val, y_reg_train, y_reg_val = train_test_split(
        X_temp, y_clf_temp, y_reg_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_clf_temp
    )

    return (X_train, X_val, X_test, 
            y_clf_train, y_clf_val, y_clf_test, 
            y_reg_train, y_reg_val, y_reg_test)

def load_feature_store_metadata():
    """
    Loads feature store schema & default values.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fp = os.path.join(base_dir, 'data', 'feature_store.json')
    if os.path.exists(fp):
        with open(fp, 'r') as f:
            return json.load(f)
    return {}

import pandas as pd
import pytest
import yaml
import os

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

@pytest.fixture
def raw_data():
    params = load_params()
    return pd.read_csv(params['data']['raw_file'])

def test_columns_exist(raw_data):
    required_columns = ['Crop', 'Season', 'State', 'Yield', 'Annual_Rainfall', 'Pesticide']
    for col in required_columns:
        assert col in raw_data.columns, f"Column {col} missing"

def test_no_nulls_in_critical_columns(raw_data):
    critical_cols = ['Yield', 'Annual_Rainfall', 'Pesticide']
    for col in critical_cols:
        assert raw_data[col].isnull().sum() == 0, f"Nulls found in {col}"

def test_yield_positive(raw_data):
    assert (raw_data['Yield'] >= 0).all(), "Negative Yield values found"

def test_rainfall_positive(raw_data):
    assert (raw_data['Annual_Rainfall'] >= 0).all(), "Negative Rainfall values found"

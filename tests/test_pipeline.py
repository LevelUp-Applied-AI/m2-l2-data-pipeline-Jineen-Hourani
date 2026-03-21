"""
Lab 2 — Learner Test File

Write your own pytest tests here. You must implement at least 3 test functions:
  - test_load_data_returns_dataframe
  - test_clean_data_no_nulls
  - test_add_features_creates_revenue

The autograder will run your tests as part of the CI check.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import pytest
from pipeline import load_data, clean_data, add_features


# ─── Test 1 ───────────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe():
    """load_data should return a DataFrame with expected columns and rows."""
    # Call load_data('data/sales_records.csv')
    df = load_data('data/sales_records.csv')
    # Assert the result is a pd.DataFrame
    assert isinstance(df, pd.DataFrame)
    # Assert len(df) > 0
    assert len(df) > 0
    # Assert all expected columns are present:
    #        'date', 'store_id', 'product_category', 'quantity', 'unit_price', 'payment_method'
    expected_cols = ['date', 'store_id', 'product_category', 'quantity', 'unit_price', 'payment_method']
    for col in expected_cols:
        assert col in df.columns


# ─── Test 2 ───────────────────────────────────────────────────────────────────

def test_clean_data_no_nulls():
    """After clean_data, quantity and unit_price should have no NaN values."""
    # Load the data, then call clean_data
    df = load_data('data/sales_records.csv')
    # Assert cleaned['quantity'].isna().sum() == 0
    cleaned = clean_data(df)
    assert cleaned['quantity'].isna().sum() == 0
    # Assert cleaned['unit_price'].isna().sum() == 0
    assert cleaned['unit_price'].isna().sum() == 0


# ─── Test 3 ───────────────────────────────────────────────────────────────────

def test_add_features_creates_revenue():
    """add_features should add a 'revenue' column equal to quantity * unit_price."""
    # Load and clean the data, then call add_features
    df = load_data('data/sales_records.csv')
    cleaned = clean_data(df)
    enriched = add_features(cleaned)
    # Assert 'revenue' in df.columns
    assert 'revenue' in enriched.columns
    # TODO: Assert the revenue values equal quantity * unit_price
    #        Use pd.testing.assert_series_equal for float comparison
    expected_revenue = enriched['quantity'] * enriched['unit_price']
    pd.testing.assert_series_equal(enriched['revenue'], expected_revenue, check_names=False)
    

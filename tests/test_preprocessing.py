# tests/test_preprocessing.py
import pandas as pd
import pytest
from src.preprocessing import build_label, build_features, LEAK_COLS

"""
builds a sample dataframe to use in test cases
""" 
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "FIPS": ["001", "002", "003", "004"],
        "COUNTY": ["A", "B", "C", "D"],
        "ST_ABBR": ["PA", "PA", "OH", "OH"],
        "RPL_THEMES": [0.90, 0.10, 0.80, 0.20],
        "RPL_THEME1": [0.85, 0.15, 0.75, 0.25],
        "RPL_THEME2": [0.80, 0.20, 0.70, 0.30],
        "RPL_THEME3": [0.88, 0.12, 0.78, 0.22],
        "RPL_THEME4": [0.75, 0.25, 0.65, 0.35],
        "ANNUAL_CHECKUP": [10, 90, 15, 85],
        "MAMMOGRAPHY": [20, 80, 25, 75],
        "RURAL_CODE": [5, 1, 4, 2],
        "EP_POV150": [30, 5, 25, 8],
        "EP_MINRTY": [40, 10, 35, 12],
    })


"""
ensures [build_label] (label engineering) sucessfully 
returns a dataframe of binary labels  
"""
def test_build_label_returns_binary(sample_df):
    labels = build_label(sample_df)
    assert set(labels.unique()).issubset({0, 1})


"""
check if [build_label] sucessfully flags high and low risk 
county A: high SVI, low checkup, low mammography, rural -> should be high risk
county B: low SVI, high checkup/mammography, urban -> should be low risk
"""
def test_build_label_risk_flags(sample_df):
    labels = build_label(sample_df)
    assert labels.iloc[0] == 1
    assert labels.iloc[1] == 0


"""
ensures that [build_features] correctly drops 
leak columns 
"""
def test_build_features_drops_leak_cols(sample_df):
    sample_df = sample_df.copy()
    sample_df["UNDERDIAGNOSIS_RISK"] = build_label(sample_df)  

    X = build_features(sample_df)
    leaked = [col for col in LEAK_COLS if col in X.columns]
    assert leaked == [], f"Leak columns still present: {leaked}"


"""
ensures that [build_features] can create an X with a given list of 
specific feature_cols that were found through our random forest 
model 
"""
def test_build_features_with_specific_feature_list(sample_df):
    X = build_features(sample_df, feature_cols=["EP_POV150", "EP_MINRTY"])
    assert list(X.columns) == ["EP_POV150", "EP_MINRTY"]
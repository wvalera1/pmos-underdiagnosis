"""
Feature preparation for the PMOS underdiagnosis risk model

Includes columns used/correlated with those used in label engineering
that must be dropped to prevent label leakage 
(RPL_THEMES, ANNUAL_CHECKUP, MAMMOGRAPHY, and RURAL_CODE where used
to construct UNDERDIAGNOSIS_RISK)
"""

import pandas as pd

LEAK_COLS = ['UNDERDIAGNOSIS_RISK', 'FIPS', 'COUNTY', 'ST_ABBR',
             'RPL_THEMES', 'ANNUAL_CHECKUP', 'MAMMOGRAPHY', 'RURAL_CODE',
             'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4']


def build_label(df: pd.DataFrame) -> pd.Series: 
    """
    [build_label] constructs the composite binary label
    Returns [UNDERDIAGNOSIS_RISK] a df of binary labels

    No direct data on county-level PMOS underdiagnosis in any dataset
    Approach: Construct a composite binary label from three "signals" grounded in literature
    ICD rates excluded due to underestimation of true PMOS prevalence (Neven)

    Signal 1: High SVI (Silva)
    Signal 2: Low preventative healthcare engagement (Silva)
    Signal 3: High NCHS code (Ramphul)

    Composite Label: Signal 1 AND (Signal 2 OR 3)
        Two potential pathways to define underdiagnosis:
            A: Vulnerable + disengaged from healthcare (Silva)
            B: Vulnerable + geographically isolated (Ramphul)
    """

    # Signal 1 - in top-quartile social vulnerability
    svi_high = df["RPL_THEMES"] >= df["RPL_THEMES"].quantile(0.75)

    # Signal 2 - low preventative healthcare engagement
    # Cervical screening (CERVICAL) unavailabile in PLACES 2025
    # due to survey question change, replaced with mammography
    checkup_col = "ANNUAL_CHECKUP"
    mammography_col = "MAMMOGRAPHY"

    preventative_low = (
        (df[checkup_col] <= df[checkup_col].quantile(0.25)) |
        (df[mammography_col] <= df[mammography_col].quantile(0.25))
    )

    # Signal 3: rural/periurban area (NCHS code >= 4)
    rural_high = df["RURAL_CODE"] >= 4

    # Composite binary label - returns 0 (low risk) or 1 (high risk)
    df["UNDERDIAGNOSIS_RISK"] = (
        svi_high & (preventative_low | rural_high)
    ).astype(int)

    # Check distribution, aim for 20-35% positive rate
    counts = df["UNDERDIAGNOSIS_RISK"].value_counts()
    positive_rate = counts[1] / len(df) * 100
    print(f"High-risk counties (label = 1): {counts[1]}")
    print(f"Low-risk counties (label = 0): {counts[0]}")
    print(f"Positive rate: {positive_rate:.1f}%")

    return (svi_high & (preventative_low | rural_high)).astype(int)

def build_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """
    [build_features] builds feature matrix X (which drops all leaked columns)
    """
    
    X = df.drop(columns=LEAK_COLS, errors="ignore") # catches errors for missing LEAK_COLS in csv
    if feature_cols is not None: 
        X = X[feature_cols]
    return X


"""
Top 10 features selected through Random Forest importance ranking
used for Logistic Regression model 
"""
TOP_10_FEATURES = [
    "EP_POV150", "EP_MINRTY", "EP_NOHSDP", "DIABETES_PREV", 
    "EP_HBURD", "EP_UNEMP", "FOOD_INSECURITY", "EP_HISP", 
    "EP_NOVEH", "EP_MUNIT"
]
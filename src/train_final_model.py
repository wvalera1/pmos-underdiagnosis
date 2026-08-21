"""
[main] trains the final XGBoost model on all avaliable labeled data 
and saves it for use in the Streamlit app

Usage: 
    python -m src.train_final_model
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve

from src.preprocessing import build_label, build_features, TOP_10_FEATURES

DATA_PATH = "data/processed/cleaned_merged.csv"
OUTPUT_CSV_PATH = "data/predictions/pmos_predictions_all_counties.csv"
N_ESTIMATORS = 372

def main(): 
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    df["UNDERDIAGNOSIS_RISK"] = build_label(df)

    X = build_features(df)
    y = df["UNDERDIAGNOSIS_RISK"]
    spw = (y == 0).sum() / (y == 1).sum()

    final_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # recompute on full y
        eval_metric="logloss",
        random_state=42
    )
    final_model.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = cross_val_predict(final_model, X, y, cv=cv, method="predict_proba")[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, all_probs)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-10)
    final_threshold = float(thresholds[np.argmax(f2)])

    print(f"Final threshold: {final_threshold: .3f}")

    # generate predictions for every county and save as CSV
    final_probs = final_model.predict_proba(X)[:, 1]
    df["RISK_PROBABILITY"] = final_probs
    df["PREDICTED_RISK"] = (final_probs >= final_threshold).astype(int)
    df["RISK_LABEL"] = df["PREDICTED_RISK"].map({1: "High Risk", 0: "Low Risk"})
    df["TRUE_LABEL"] = y.values
    df["label"] = df["COUNTY"] + ", " + df["ST_ABBR"]

    output_cols = ["FIPS", "COUNTY", "ST_ABBR", "RISK_PROBABILITY", "PREDICTED_RISK", "TRUE_LABEL", "label"]
    df[output_cols].to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_CSV_PATH}")


if __name__ == "__main__": 
    main()
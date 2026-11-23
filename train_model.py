"""
Train the DiaBD diabetes-risk model and save it to model.joblib.

This mirrors the baseline pipeline in Diabetes_Risk_Prediction.ipynb:
  1. encode gender / target
  2. replace biologically impossible zeros with the column mean
  3. 80/20 stratified split (random_state=42)
  4. StandardScaler + LogisticRegression(liblinear)

Run once (and again whenever you change the data or model):
    python train_model.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
CSV_PATH = HERE / "data" / "diabd.csv"
MODEL_PATH = HERE / "model.joblib"

ZERO_AS_MISSING = ["systolic_bp", "diastolic_bp", "glucose", "height", "weight", "bmi"]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["diabetic"] = (df["diabetic"] == "Yes").astype(int)

    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, df.loc[df[col] != 0, col].mean())

    X, y = df.drop(columns="diabetic"), df["diabetic"]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, proba > 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": features,
            # Mean of each feature (used when the user leaves a field empty)
            "means": dict(zip(features, scaler.mean_.tolist())),
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    print(f"Saved {MODEL_PATH.name}  |  {metrics}")


if __name__ == "__main__":
    main()

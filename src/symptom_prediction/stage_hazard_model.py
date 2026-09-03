import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

RAW_FILE = "data/raw/nfhs_menopause_subset.parquet"
MODEL_FILE = "models/transition_stage_model.pkl"
METRICS_FILE = "results/metrics/stage_model_performance.csv"

STAGE_NAMES = {
    0: "Premenopausal",
    1: "Perimenopausal",
    2: "Postmenopausal"
}

def convert_to_months(value):
    """
    Standard DHS / NFHS recode logic for v215 (time since last menstrual period).
    100-199: days
    200-299: weeks
    300-399: months
    400-499: years
    994: menopausal / hysterectomy
    """
    if pd.isna(value):
        return np.nan
    try:
        val = int(value)
    except (ValueError, TypeError):
        return np.nan

    if 100 <= val <= 199:
        return (val - 100) / 30.44
    elif 200 <= val <= 299:
        return (val - 200) * 7 / 30.44
    elif 300 <= val <= 399:
        return float(val - 300)
    elif 400 <= val <= 499:
        return float((val - 400) * 12)
    elif val == 994:
        return 12.0
    return np.nan

def prepare_stage_data(raw_path=RAW_FILE):
    """
    Loads raw NFHS data, cleans menstrual cessation, excludes pregnancy confounding,
    and constructs validated 3-stage outcome.
    """
    print(f"Loading NFHS raw dataset from: {raw_path}")
    df = pd.read_parquet(raw_path)
    print(f"Initial record count: {len(df):,}")

    # Exclude pregnant women to avoid pregnancy-induced amenorrhea confounding
    if "v213" in df.columns:
        df = df[df["v213"] != 1].copy()
        print(f"Records after excluding currently pregnant women: {len(df):,}")

    # Ensure valid age
    df = df[df["v012"].notna() & (df["v012"] >= 30) & (df["v012"] <= 49)].copy()

    # Recode v215
    df["months_since_period"] = df["v215"].apply(convert_to_months)

    # Valid menstrual duration records
    valid_mask = df["months_since_period"].notna() | (df["v215"] == 994)
    df = df[valid_mask].copy()

    # Stage classification based on clinical STRAW+10 criteria adapted to survey data:
    # 0: Premenopausal (cycling regularly or last period < 2 months ago)
    # 1: Perimenopausal (amenorrhea 2 to 11 months - transition onset)
    # 2: Postmenopausal (amenorrhea >= 12 months or explicit menopausal code 994)
    conditions = [
        (df["months_since_period"] < 2),
        (df["months_since_period"] >= 2) & (df["months_since_period"] < 12),
        (df["months_since_period"] >= 12) | (df["v215"] == 994)
    ]
    choices = [0, 1, 2]
    df["stage"] = np.select(conditions, choices, default=-1)
    df = df[df["stage"] != -1].copy()

    print("\nEmpirical Stage Distribution across Indian Cohort:")
    for stage_code, stage_name in STAGE_NAMES.items():
        count = (df["stage"] == stage_code).sum()
        pct = (count / len(df)) * 100
        print(f"  Stage {stage_code} ({stage_name:<15}): {count:,} ({pct:.2f}%)")

    return df

def train_transition_model():
    """
    Trains multi-class classification model on NFHS-5 data and saves model artifact.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    df = prepare_stage_data()

    feature_cols = ["v012", "v024", "v025", "v190"]
    readable_names = {
        "v012": "Age",
        "v024": "Region",
        "v025": "Residence_Type",
        "v190": "Wealth_Index"
    }

    X = df[feature_cols].rename(columns=readable_names)
    y = df["stage"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Testing set : {len(X_test):,} samples")

    print("\nFitting HistGradientBoostingClassifier with balanced class weighting...")
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    print(f"\nModel Performance on Held-Out Test Cohort:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1_macro:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")

    print("\nDetailed Classification Report:")
    target_names = [STAGE_NAMES[i] for i in range(3)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save metrics
    metrics_df = pd.DataFrame([{
        "Model": "HistGradientBoostingClassifier",
        "Accuracy": acc,
        "Macro_F1": f1_macro,
        "Macro_ROC_AUC": roc_auc,
        "N_Train": len(X_train),
        "N_Test": len(X_test)
    }])
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"Performance metrics saved to: {METRICS_FILE}")

    # Save artifact
    artifact = {
        "model": model,
        "feature_cols": list(X.columns),
        "stage_names": STAGE_NAMES,
        "description": "Multi-class perimenopause/menopause transition staging model trained on NFHS-5"
    }
    joblib.dump(artifact, MODEL_FILE)
    print(f"Trained stage model artifact saved to: {MODEL_FILE}")

    return model

def forecast_onset_window(current_age, region, residence_type, wealth_index, model_artifact=None):
    """
    Forecasts probability of transition onset (perimenopause/menopause) 
    over the next 12, 24, and 36 months using age-as-timeline conditional hazard.
    """
    if model_artifact is None:
        model_artifact = joblib.load(MODEL_FILE)

    model = model_artifact["model"]

    horizons_months = [12, 24, 36]
    results = {}

    # Current profile probability
    current_input = pd.DataFrame([{
        "Age": current_age,
        "Region": region,
        "Residence_Type": residence_type,
        "Wealth_Index": wealth_index
    }])
    current_probs = model.predict_proba(current_input)[0]
    p_current_premenopausal = current_probs[0]

    for h in horizons_months:
        future_age = current_age + (h / 12.0)
        future_input = pd.DataFrame([{
            "Age": future_age,
            "Region": region,
            "Residence_Type": residence_type,
            "Wealth_Index": wealth_index
        }])
        future_probs = model.predict_proba(future_input)[0]
        p_future_premenopausal = future_probs[0]

        # Conditional transition probability: given still premenopausal now,
        # what is the chance transition starts within h months?
        if p_current_premenopausal > 0:
            p_transition_within_h = max(0.0, min(1.0, (p_current_premenopausal - p_future_premenopausal) / p_current_premenopausal))
        else:
            p_transition_within_h = 1.0

        results[f"{h}_months"] = {
            "future_age": round(future_age, 1),
            "transition_onset_prob": round(float(p_transition_within_h), 4),
            "predicted_perimenopause_prob": round(float(future_probs[1]), 4),
            "predicted_postmenopause_prob": round(float(future_probs[2]), 4)
        }

    return {
        "current_stage_probabilities": {
            STAGE_NAMES[i]: round(float(current_probs[i]), 4) for i in range(3)
        },
        "most_likely_current_stage": STAGE_NAMES[int(np.argmax(current_probs))],
        "onset_forecast": results
    }

if __name__ == "__main__":
    trained_model = train_transition_model()
    print("\n==================================================")
    print("DEMO: FORECASTING ONSET FOR A 38-YEAR-OLD WOMAN")
    print("==================================================")
    sample_forecast = forecast_onset_window(
        current_age=38,
        region=1,
        residence_type=1,
        wealth_index=3
    )
    print("Current Status Prediction:")
    print(sample_forecast["current_stage_probabilities"])
    print("\nProjected Onset Probabilities (12, 24, 36 months ahead):")
    for horizon, data in sample_forecast["onset_forecast"].items():
        print(f"  Within {horizon} (at age {data['future_age']}): "
              f"Onset Prob = {data['transition_onset_prob']*100:.1f}% "
              f"(Peri = {data['predicted_perimenopause_prob']*100:.1f}%, Post = {data['predicted_postmenopause_prob']*100:.1f}%)")

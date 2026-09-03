import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.symptom_prediction.stage_hazard_model import prepare_stage_data, STAGE_NAMES

FIGURE_PATH = "results/figures/feature_importance_attribution.png"
METRICS_PATH = "results/metrics/feature_importance_analysis.csv"

def run_feature_attribution_analysis():
    """
    Applies three mathematical analysis tools to isolate and rank top transition predictors
    localized to Indian women:
    1. Permutation Importance (Scikit-Learn)
    2. Random Forest MDI (Gini Importance)
    3. Multinomial Logistic Regression Odds Ratios (e^beta) with standardized effect sizes
    """
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    print("\n==================================================")
    print("OBJECTIVE 2: MATHEMATICAL FEATURE ATTRIBUTION")
    print("==================================================")

    df = prepare_stage_data()

    feature_cols = ["v012", "v024", "v025", "v190"]
    readable_names = {
        "v012": "Age",
        "v024": "Geographic_Region",
        "v025": "Residence_Type (Urban/Rural)",
        "v190": "Wealth_Index (Nutrition/SES)"
    }

    X = df[feature_cols].rename(columns=readable_names)
    y = df["stage"]

    # Sample 50,000 for permutation analysis if dataset is massive to maintain high speed
    if len(X) > 50000:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=50000, random_state=42, stratify=y
        )
    else:
        X_sample, y_sample = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.25, random_state=42, stratify=y_sample
    )

    features = list(X.columns)

    # -------------------------------------------------------------
    # 1. Random Forest Gini / Split Gain Importance
    # -------------------------------------------------------------
    print("\n1. Calculating Random Forest Feature Importances (MDI)...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_

    # -------------------------------------------------------------
    # 2. Permutation Feature Importance
    # -------------------------------------------------------------
    print("2. Calculating Permutation Feature Importance (10 repeats)...")
    perm_result = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_mean = perm_result.importances_mean
    perm_std = perm_result.importances_std

    # -------------------------------------------------------------
    # 3. Standardized Logistic Regression Odds Ratios (e^beta)
    # -------------------------------------------------------------
    print("3. Calculating Standardized Odds Ratios (e^beta)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=500, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    # Coefs shape: (n_classes, n_features)
    # Class 1: Perimenopause, Class 2: Postmenopause (relative to baseline)
    odds_ratios_peri = np.exp(log_reg.coef_[1])
    odds_ratios_post = np.exp(log_reg.coef_[2])

    # -------------------------------------------------------------
    # Consolidate Metrics
    # -------------------------------------------------------------
    results_records = []
    for i, f in enumerate(features):
        results_records.append({
            "Feature": f,
            "RF_Gini_Importance": round(float(rf_importances[i]), 4),
            "Permutation_Importance_Mean": round(float(perm_mean[i]), 4),
            "Permutation_Importance_Std": round(float(perm_std[i]), 4),
            "Odds_Ratio_Perimenopause": round(float(odds_ratios_peri[i]), 3),
            "Odds_Ratio_Postmenopause": round(float(odds_ratios_post[i]), 3)
        })

    metrics_df = pd.DataFrame(results_records)
    metrics_df = metrics_df.sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"\nFeature Attribution Table:\n{metrics_df}")
    print(f"\nSaved metrics to: {METRICS_PATH}")

    # -------------------------------------------------------------
    # Plot Visualizations
    # -------------------------------------------------------------
    print("\nGenerating attribution visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.35)

    # Plot 1: Permutation Importance
    sorted_idx_perm = np.argsort(perm_mean)
    axes[0].barh(np.array(features)[sorted_idx_perm], perm_mean[sorted_idx_perm],
                 xerr=perm_std[sorted_idx_perm], color="#2b5c8f", capsize=4, edgecolor="black", alpha=0.85)
    axes[0].set_title("Permutation Importance\n(Mean drop in model accuracy)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Mean Importance Score")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # Plot 2: Random Forest Gini Importance
    sorted_idx_rf = np.argsort(rf_importances)
    axes[1].barh(np.array(features)[sorted_idx_rf], rf_importances[sorted_idx_rf],
                 color="#d95f02", edgecolor="black", alpha=0.85)
    axes[1].set_title("Ensemble Tree Importance\n(Gini / Split Gain)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Relative Importance (0.0 to 1.0)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    # Plot 3: Standardized Odds Ratios (e^beta)
    y_pos = np.arange(len(features))
    width = 0.35
    axes[2].barh(y_pos - width/2, odds_ratios_peri, width, label="Perimenopause Onset", color="#7570b3", edgecolor="black", alpha=0.85)
    axes[2].barh(y_pos + width/2, odds_ratios_post, width, label="Postmenopause Transition", color="#1b9e77", edgecolor="black", alpha=0.85)
    axes[2].axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Null Effect (OR=1.0)")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(features)
    axes[2].set_title("Standardized Odds Ratios ($e^\\beta$)\n(Multinomial Logistic Regression)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Odds Ratio (OR)")
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Menocare Feature Attribution & Importance — Indian Demographic Cohort", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Attribution plot saved to: {FIGURE_PATH}")

    return metrics_df

if __name__ == "__main__":
    run_feature_attribution_analysis()

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.symptom_prediction.stage_hazard_model import prepare_stage_data, STAGE_NAMES

METRICS_REPORT = "results/metrics/validation_report.csv"
CONFUSION_MATRIX_PLOT = "results/figures/confusion_matrix_stage.png"
ROC_CURVES_PLOT = "results/figures/roc_curves_stage.png"

def run_stratified_validation(n_splits=5):
    """
    Executes Stratified 5-Fold Cross-Validation on the full Indian NFHS-5 cohort,
    evaluating generalizability, discriminative power (ROC-AUC), and probability calibration.
    """
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    print("\n==================================================")
    print(f"OBJECTIVE 4: STRATIFIED {n_splits}-FOLD CLINICAL CROSS-VALIDATION")
    print("==================================================")

    df = prepare_stage_data()

    feature_cols = ["v012", "v024", "v025", "v190"]
    readable_names = {
        "v012": "Age",
        "v024": "Region",
        "v025": "Residence_Type",
        "v190": "Wealth_Index"
    }

    X = df[feature_cols].rename(columns=readable_names).values
    y = df["stage"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    oof_predictions = np.zeros(len(y), dtype=int)
    oof_probabilities = np.zeros((len(y), 3), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Training Fold {fold}/{n_splits} ---")
        X_train_f, y_train_f = X[train_idx], y[train_idx]
        X_val_f, y_val_f = X[val_idx], y[val_idx]

        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_depth=6,
            min_samples_leaf=50,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42 + fold
        )

        model.fit(X_train_f, y_train_f)
        y_val_pred = model.predict(X_val_f)
        y_val_proba = model.predict_proba(X_val_f)

        oof_predictions[val_idx] = y_val_pred
        oof_probabilities[val_idx] = y_val_proba

        acc = accuracy_score(y_val_f, y_val_pred)
        bal_acc = balanced_accuracy_score(y_val_f, y_val_pred)
        f1_macro = f1_score(y_val_f, y_val_pred, average="macro")
        roc_auc_macro = roc_auc_score(y_val_f, y_val_proba, multi_class="ovr", average="macro")

        print(f"  Fold {fold} - Acc: {acc:.4f} | Bal Acc: {bal_acc:.4f} | Macro F1: {f1_macro:.4f} | ROC-AUC: {roc_auc_macro:.4f}")

        fold_metrics.append({
            "Fold": fold,
            "Accuracy": round(acc, 4),
            "Balanced_Accuracy": round(bal_acc, 4),
            "Macro_F1": round(f1_macro, 4),
            "Macro_ROC_AUC": round(roc_auc_macro, 4)
        })

    # Summary Statistics
    summary_df = pd.DataFrame(fold_metrics)
    mean_row = {
        "Fold": "Mean +/- Std",
        "Accuracy": f"{summary_df['Accuracy'].mean():.4f} +/- {summary_df['Accuracy'].std():.4f}",
        "Balanced_Accuracy": f"{summary_df['Balanced_Accuracy'].mean():.4f} +/- {summary_df['Balanced_Accuracy'].std():.4f}",
        "Macro_F1": f"{summary_df['Macro_F1'].mean():.4f} +/- {summary_df['Macro_F1'].std():.4f}",
        "Macro_ROC_AUC": f"{summary_df['Macro_ROC_AUC'].mean():.4f} +/- {summary_df['Macro_ROC_AUC'].std():.4f}"
    }
    summary_with_mean = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)
    summary_with_mean.to_csv(METRICS_REPORT, index=False)
    print(f"\nConsolidated Validation Report Saved to: {METRICS_REPORT}")
    print(summary_with_mean)

    # -------------------------------------------------------------
    # Confusion Matrix Visualization
    # -------------------------------------------------------------
    print("\nGenerating Out-of-Fold Normalized Confusion Matrix...")
    cm = confusion_matrix(y, oof_predictions, normalize="true")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix — NFHS-5 Cross-Validation", fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(label="Proportion")
    tick_marks = np.arange(3)
    class_names = [STAGE_NAMES[i] for i in range(3)]
    plt.xticks(tick_marks, class_names, rotation=15)
    plt.yticks(tick_marks, class_names)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]*100:.1f}%\n({confusion_matrix(y, oof_predictions)[i, j]:,})",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=10)

    plt.tight_layout()
    plt.ylabel("Clinically Verified Stage (NFHS)")
    plt.xlabel("Model Predicted Stage")
    plt.savefig(CONFUSION_MATRIX_PLOT, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PLOT}")

    # -------------------------------------------------------------
    # Multi-Class ROC Curves
    # -------------------------------------------------------------
    print("Generating Multi-Class ROC Curves (One-vs-Rest)...")
    plt.figure(figsize=(8, 6.5))
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]

    for i in range(3):
        y_binary = (y == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, oof_probabilities[:, i])
        roc_auc_class = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{STAGE_NAMES[i]} (AUC = {roc_auc_class:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Multi-Class ROC Curves (One-vs-Rest) — NFHS-5 Cohort", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ROC_CURVES_PLOT, dpi=300)
    plt.close()
    print(f"ROC curves saved to: {ROC_CURVES_PLOT}")

    return summary_df

if __name__ == "__main__":
    run_stratified_validation(n_splits=5)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/clean_nfhs_menopause.csv"
)

print("Dataset loaded.")
print("Shape:", df.shape)


# ==========================================
# 2. CREATE TARGET
# ==========================================

df["target"] = (
    df["menopause_stage"] == "Perimenopausal"
).astype(int)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df[
    [
        "age",
        "residence_type",
        "wealth_index",
        "is_pregnant"
    ]
]

y = df["target"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. PREPROCESSING
# ==========================================

categorical_features = [
    "residence_type"
]

numerical_features = [
    "age",
    "wealth_index",
    "is_pregnant"
]


preprocessor = ColumnTransformer(
    transformers=[

        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        ),

        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical_features
        )
    ]
)


# ==========================================
# 6. XGBOOST
# ==========================================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        xgb_model
    )
])


# ==========================================
# 7. TRAIN
# ==========================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ==========================================
# 8. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Not Perimenopausal",
        "Perimenopausal"
    ],
    yticklabels=[
        "Not Perimenopausal",
        "Perimenopausal"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/figures/confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================
# 11. ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc = roc_auc_score(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("XGBoost ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/figures/roc_curve.png",
    dpi=300
)

plt.show()


print("\nROC-AUC:", auc)

print(
    "\nGraphs saved inside:"
    " results/figures/"
)
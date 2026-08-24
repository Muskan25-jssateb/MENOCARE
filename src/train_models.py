import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


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
# 3. SELECT FEATURES
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


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


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
# 6. XGBOOST MODEL
# ==========================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
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
# 9. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# 10. RESULTS
# ==========================================

print("\n================================")
print("XGBOOST RESULTS")
print("================================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
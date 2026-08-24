import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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
# 6. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}


# ==========================================
# 7. STORE RESULTS
# ==========================================

results = []


# ==========================================
# 8. TRAIN EACH MODEL
# ==========================================

for name, classifier in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ])

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predictions
    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
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

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })


# ==========================================
# 9. CREATE RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n\n==============================================")
print("             MODEL COMPARISON")
print("==============================================")

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 11. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "results/metrics/model_comparison.csv",
    index=False
)

print(
    "\nResults saved to:"
    " results/metrics/model_comparison.csv"
)
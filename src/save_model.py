import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv(
    "data/processed/clean_nfhs_menopause.csv"
)


# Target
df["target"] = (
    df["menopause_stage"] == "Perimenopausal"
).astype(int)


# Features
X = df[
    [
        "age",
        "residence_type",
        "wealth_index",
        "is_pregnant"
    ]
]

y = df["target"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            SimpleImputer(strategy="median"),
            [
                "age",
                "wealth_index",
                "is_pregnant"
            ]
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
            ["residence_type"]
        )
    ]
)


# XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# Complete pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb)
])


# Train on training data
print("Training final model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# Save model
joblib.dump(
    model,
    "models/menopause_classifier.pkl"
)

print(
    "\nModel saved to:"
    " models/menopause_classifier.pkl"
)
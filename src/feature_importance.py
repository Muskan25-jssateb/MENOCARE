import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
# 3. FEATURES
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


model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        xgb
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
# 8. GET FEATURE NAMES
# ==========================================

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


# ==========================================
# 9. GET FEATURE IMPORTANCE
# ==========================================

importance = (
    model
    .named_steps["classifier"]
    .feature_importances_
)


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
# 10. PRINT RESULTS
# ==========================================

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

print(
    importance_df.to_string(index=False)
)


# ==========================================
# 11. SAVE RESULTS
# ==========================================

importance_df.to_csv(
    "results/metrics/feature_importance.csv",
    index=False
)


# ==========================================
# 12. PLOT
# ==========================================

plt.figure(figsize=(8, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "XGBoost Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "results/figures/feature_importance.png",
    dpi=300
)

plt.show()


print(
    "\nFeature importance saved to:"
    " results/metrics/feature_importance.csv"
)

print(
    "Graph saved to:"
    " results/figures/feature_importance.png"
)

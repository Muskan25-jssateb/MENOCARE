import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor


# ==========================================
# FILES
# ==========================================

RAW_FILE = "data/raw/nfhs_menopause_subset.parquet"
TARGET_FILE = "data/processed/menopause_span_by_region.csv"
RESULT_FILE = "results/metrics/menopause_span_models.csv"
MODEL_FILE = "models/menopause_span_model.pkl"


# ==========================================
# CREATE OUTPUT DIRECTORIES
# ==========================================

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_parquet(RAW_FILE)
target = pd.read_csv(TARGET_FILE)


# ==========================================
# CREATE REGION-LEVEL FEATURES
# ==========================================

features = (
    df.groupby("v024")
    .agg(
        Median_Age=("v012", "median"),
        Mean_Age=("v012", "mean"),

        # v025: 1 = Urban, 2 = Rural
        Urban_Rate=("v025", lambda x: (x == 1).mean()),

        Mean_Wealth=("v190", "mean"),

        # v213: 1 = Currently pregnant, 0 = No
        Pregnancy_Rate=("v213", "mean"),

        Sample_Size=("v012", "count")
    )
    .reset_index()
)


# ==========================================
# MERGE TARGET
# ==========================================

data = features.merge(
    target[["v024", "Menopause_Span"]],
    on="v024",
    how="inner"
)


# ==========================================
# FEATURES
# ==========================================

feature_columns = [
    "Median_Age",
    "Mean_Age",
    "Urban_Rate",
    "Mean_Wealth",
    "Pregnancy_Rate",
    "Sample_Size"
]

X = data[feature_columns]
y = data["Menopause_Span"]


# ==========================================
# DISPLAY CLEAN DATA SUMMARY
# ==========================================

print("\n========================================")
print("MENOPAUSE SPAN MODEL TRAINING")
print("========================================")

print(f"Regions available : {len(data)}")
print(f"Features used     : {len(feature_columns)}")
print(f"Target            : Menopause_Span")


# ==========================================
# MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
}


# ==========================================
# LEAVE-ONE-REGION-OUT VALIDATION
# ==========================================

loo = LeaveOneOut()

results = []


for model_name, model in models.items():

    actual = []
    predicted = []

    for train_index, test_index in loo.split(X):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(X_test)[0]

        actual.append(y_test.iloc[0])
        predicted.append(prediction)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ==========================================
# MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE"
).reset_index(drop=True)


print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

for _, row in results_df.iterrows():

    print(
        f"{row['Model']:<20} "
        f"MAE: {row['MAE']:.2f} years   "
        f"RMSE: {row['RMSE']:.2f} years   "
        f"R²: {row['R2']:.3f}"
    )


# ==========================================
# SAVE MODEL COMPARISON
# ==========================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ==========================================
# SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = models[best_model_name]

best_model.fit(
    X,
    y
)


# ==========================================
# SAVE FINAL MODEL
# ==========================================

joblib.dump(
    best_model,
    MODEL_FILE
)


print("\n========================================")
print("FINAL MODEL")
print("========================================")

print(f"Selected model : {best_model_name}")
print(f"Model saved    : {MODEL_FILE}")
print(f"Results saved  : {RESULT_FILE}")
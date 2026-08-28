import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ==========================================
# FILES
# ==========================================

RAW_FILE = "data/raw/nfhs_menopause_subset.parquet"
TARGET_FILE = "data/processed/menopause_span_by_region.csv"

RESULT_FILE = "results/metrics/menopause_span_loocv.csv"
GRAPH_FILE = "results/figures/menopause_span_loocv.png"


# ==========================================
# CREATE OUTPUT DIRECTORIES
# ==========================================

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_parquet(RAW_FILE)
target = pd.read_csv(TARGET_FILE)


# ==========================================
# CREATE REGION FEATURES
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
# FEATURES AND TARGET
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
# LEAVE-ONE-REGION-OUT VALIDATION
# ==========================================

loo = LeaveOneOut()

actual = []
predicted = []
regions = []


for train_index, test_index in loo.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )[0]

    actual.append(
        y_test.iloc[0]
    )

    predicted.append(
        prediction
    )

    regions.append(
        int(
            data.iloc[test_index[0]]["v024"]
        )
    )


# ==========================================
# METRICS
# ==========================================

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


# ==========================================
# SAVE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Region": regions,
    "Actual_Span": actual,
    "Predicted_Span": predicted
})

results["Absolute_Error"] = (
    results["Predicted_Span"] -
    results["Actual_Span"]
).abs()

results.to_csv(
    RESULT_FILE,
    index=False
)


# ==========================================
# CLEAN OUTPUT
# ==========================================

print("\n========================================")
print("MENOPAUSE SPAN MODEL VALIDATION")
print("========================================")

print("Validation method : Leave-One-Region-Out")
print(f"Regions evaluated : {len(results)}")
print(f"MAE               : {mae:.2f} years")
print(f"RMSE              : {rmse:.2f} years")
print(f"R²                : {r2:.3f}")

print("\nResults saved to:")
print(RESULT_FILE)


# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted
)

minimum = min(
    min(actual),
    min(predicted)
)

maximum = max(
    max(actual),
    max(predicted)
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Menopause Transition Span (Years)"
)

plt.ylabel(
    "Predicted Menopause Transition Span (Years)"
)

plt.title(
    "Leave-One-Region-Out: Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    GRAPH_FILE,
    dpi=300
)

plt.show()

print("Graph saved to:")
print(GRAPH_FILE)
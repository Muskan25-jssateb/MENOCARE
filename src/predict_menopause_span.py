import pandas as pd
import joblib


# ==========================================
# FILES
# ==========================================

RAW_FILE = "data/raw/nfhs_menopause_subset.parquet"
TARGET_FILE = "data/processed/menopause_span_by_region.csv"
MODEL_FILE = "models/menopause_span_model.pkl"


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

df = pd.read_parquet(RAW_FILE)
target = pd.read_csv(TARGET_FILE)

model = joblib.load(MODEL_FILE)


# ==========================================
# CREATE REGION FEATURES
# ==========================================

region_features = (
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
# AVAILABLE REGIONS
# ==========================================

available_regions = sorted(
    region_features["v024"].astype(int).unique()
)


print("\n========================================")
print("MENOPAUSE TRANSITION SPAN PREDICTION")
print("========================================")

print(
    "Available regions:",
    ", ".join(map(str, available_regions))
)


# ==========================================
# USER INPUT
# ==========================================

try:

    region = int(
        input("\nEnter region code: ")
    )

except ValueError:

    print("\nInvalid input. Enter a numeric region code.")
    raise SystemExit


# ==========================================
# CHECK REGION
# ==========================================

region_data = region_features[
    region_features["v024"] == region
]


if region_data.empty:

    print(f"\nRegion {region} is not available.")
    raise SystemExit


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

X = region_data[feature_columns]


# ==========================================
# PREDICTION
# ==========================================

predicted_span = model.predict(X)[0]


# ==========================================
# GET NFHS-DERIVED VALUES
# ==========================================

actual_data = target[
    target["v024"] == region
]


print("\n========================================")
print("PREDICTION RESULT")
print("========================================")

print(f"Region                         : {region}")
print(
    f"Predicted transition span     : "
    f"{predicted_span:.2f} years"
)


if not actual_data.empty:

    start_age = actual_data[
        "Median_Start_Age"
    ].iloc[0]

    end_age = actual_data[
        "Median_End_Age"
    ].iloc[0]

    actual_span = actual_data[
        "Menopause_Span"
    ].iloc[0]

    print(
        f"NFHS-derived median start age : "
        f"{start_age:.1f} years"
    )

    print(
        f"NFHS-derived median end age   : "
        f"{end_age:.1f} years"
    )

    print(
        f"NFHS-derived transition span  : "
        f"{actual_span:.1f} years"
    )


print("========================================")
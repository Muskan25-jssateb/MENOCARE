import pandas as pd
import numpy as np

INPUT_FILE = "data/raw/nfhs_menopause_subset.parquet"
OUTPUT_FILE = "data/processed/menopause_span_by_region.csv"


# ==========================================
# LOAD ORIGINAL NFHS DATA
# ==========================================

df = pd.read_parquet(INPUT_FILE)

print("Original shape:", df.shape)


# ==========================================
# CONVERT V215 TO MONTHS SINCE LAST PERIOD
# ==========================================

def convert_to_months(value):

    if pd.isna(value):
        return np.nan

    value = int(value)

    # Days
    if 100 <= value <= 199:
        days = value - 100
        return days / 30.44

    # Weeks
    if 200 <= value <= 299:
        weeks = value - 200
        return weeks * 7 / 30.44

    # Months
    if 300 <= value <= 399:
        return value - 300

    # Years
    if 400 <= value <= 499:
        years = value - 400
        return years * 12

    # Special menopause code
    if value == 994:
        return 12

    return np.nan


df["months_since_period"] = df["v215"].apply(
    convert_to_months
)


# ==========================================
# START MARKER
# 2-11 MONTHS WITHOUT PERIOD
# ==========================================

df["start_marker"] = (
    (df["months_since_period"] >= 2) &
    (df["months_since_period"] < 12)
)


# ==========================================
# END MARKER
# 12+ MONTHS WITHOUT PERIOD
# OR EXPLICIT MENOPAUSE
# ==========================================

df["end_marker"] = (
    (df["months_since_period"] >= 12) |
    (df["v215"] == 994)
)


# ==========================================
# KEEP VALID AGES
# ==========================================

df = df[
    df["v012"].notna()
]


# ==========================================
# REGION-LEVEL MEDIAN AGES
# ==========================================

start_age = (
    df[df["start_marker"]]
    .groupby("v024")["v012"]
    .median()
    .rename("Median_Start_Age")
)

end_age = (
    df[df["end_marker"]]
    .groupby("v024")["v012"]
    .median()
    .rename("Median_End_Age")
)


# ==========================================
# COMBINE
# ==========================================

result = pd.concat(
    [start_age, end_age],
    axis=1
).reset_index()


# ==========================================
# CALCULATE MENOPAUSE TRANSITION SPAN
# ==========================================

result["Menopause_Span"] = (
    result["Median_End_Age"] -
    result["Median_Start_Age"]
)


# ==========================================
# REMOVE INVALID SPANS
# ==========================================

result = result[
    result["Median_Start_Age"].notna() &
    result["Median_End_Age"].notna() &
    (result["Menopause_Span"] >= 0)
]


# ==========================================
# SAVE
# ==========================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# DISPLAY
# ==========================================

print("\n========================================")
print("MENOPAUSE SPAN BY REGION")
print("========================================")

print(result)

print("\nNumber of regions:", len(result))

print(
    "\nSaved to:",
    OUTPUT_FILE
)
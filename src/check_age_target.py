import pandas as pd


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/clean_nfhs_menopause.csv"
)

print("\n================================")
print("DATASET INFORMATION")
print("================================")

print("Shape:", df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)


# ==========================================
# SEARCH FOR POSSIBLE AGE/TIMELINE TARGETS
# ==========================================

keywords = [
    "menopause",
    "menopausal",
    "menopause_age",
    "age_menopause",
    "menopause_year",
    "last_period",
    "last_menstrual",
    "menstrual",
    "period"
]


print("\n================================")
print("POSSIBLE TARGET COLUMNS")
print("================================")

found = []

for column in df.columns:

    column_lower = column.lower()

    for keyword in keywords:

        if keyword in column_lower:

            found.append(column)
            break


if found:

    for column in found:
        print("-", column)

else:

    print(
        "No obvious menopause-age/timeline "
        "column found."
    )


# ==========================================
# DISPLAY MENOPAUSE-RELATED COLUMNS
# ==========================================

print("\n================================")
print("COLUMN DETAILS")
print("================================")

for column in found:

    print("\nColumn:", column)

    print("Data type:")
    print(df[column].dtype)

    print("Unique values:")
    print(df[column].nunique())

    print("Sample values:")
    print(
        df[column]
        .dropna()
        .head(10)
        .tolist()
    )
import pandas as pd

# Load NFHS dataset
nfhs = pd.read_csv(
    "data/processed/clean_nfhs_menopause.csv"
)

print("========== NFHS DATASET ==========")

print("Shape:")
print(nfhs.shape)

print("\nColumns:")
print(nfhs.columns.tolist())

print("\nFirst 5 rows:")
print(nfhs.head())

print("\nData types:")
print(nfhs.dtypes)

print("\nMissing values:")
print(nfhs.isnull().sum())

print("\nDuplicate rows:")
print(nfhs.duplicated().sum())

print("\nMenopause stage:")
print(
    nfhs["menopause_stage"].value_counts(
        dropna=False
    )
)
import pandas as pd

FILE = "data/raw/nfhs_menopause_subset.parquet"

print("\n========================================")
print("ORIGINAL NFHS DATA")
print("========================================")

df = pd.read_parquet(FILE)

print("Shape:", df.shape)

print("\n========================================")
print("ALL COLUMNS")
print("========================================")

for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")

print("\n========================================")
print("DISTRICT / REGION COLUMNS")
print("========================================")

keywords = [
    "district",
    "dist",
    "region",
    "state",
    "v024",
    "v025"
]

found = []

for column in df.columns:
    if any(k in column.lower() for k in keywords):
        found.append(column)
        print("-", column)

if not found:
    print("No obvious district/region column found.")

print("\n========================================")
print("MENOPAUSE / MENSTRUAL COLUMNS")
print("========================================")

keywords = [
    "menopause",
    "menopausal",
    "menstru",
    "period",
    "amenorr",
    "menses",
    "last"
]

found = []

for column in df.columns:
    if any(k in column.lower() for k in keywords):
        found.append(column)
        print("-", column)

if not found:
    print("No obvious menopause-related column found.")

print("\n========================================")
print("AGE COLUMNS")
print("========================================")

for column in df.columns:
    if "age" in column.lower():
        print("-", column)

print("\n========================================")
print("FIRST 5 ROWS")
print("========================================")

print(df.head())
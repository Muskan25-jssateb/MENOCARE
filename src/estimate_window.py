import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/clean_nfhs_menopause.csv"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "models/menopause_classifier.pkl"
)


# ==========================================
# 3. USER INFORMATION
# ==========================================

user_residence = 1
user_wealth = 3
user_pregnant = 0


# ==========================================
# 4. GET AGE RANGE FROM DATASET
# ==========================================

min_age = int(df["age"].min())
max_age = int(df["age"].max())

ages = list(
    range(min_age, max_age + 1)
)


print("\nDataset age range:")
print(f"{min_age} - {max_age}")


# ==========================================
# 5. CALCULATE PROBABILITY FOR EACH AGE
# ==========================================

results = []

for age in ages:

    sample = pd.DataFrame([
        {
            "age": age,
            "residence_type": user_residence,
            "wealth_index": user_wealth,
            "is_pregnant": user_pregnant
        }
    ])

    probability = model.predict_proba(
        sample
    )[0][1]

    results.append({
        "age": age,
        "probability": probability
    })


results_df = pd.DataFrame(results)


# ==========================================
# 6. DISPLAY AGE-WISE PROBABILITY
# ==========================================

print("\n========================================")
print("AGE-WISE PERIMENOPAUSE PROBABILITY")
print("========================================")

for _, row in results_df.iterrows():

    print(
        f"Age {int(row['age'])}: "
        f"{row['probability'] * 100:.2f}%"
    )


# ==========================================
# 7. FIND TRANSITION WINDOW
# ==========================================

threshold = 0.50

above_threshold = results_df[
    results_df["probability"] >= threshold
]


print("\n========================================")
print("ESTIMATED TRANSITION WINDOW")
print("========================================")


if len(above_threshold) > 0:

    start_age = int(
        above_threshold["age"].iloc[0]
    )

    end_age = int(
        above_threshold["age"].iloc[-1]
    )

    print(
        f"Estimated window: "
        f"{start_age}-{end_age} years"
    )

else:

    print(
        "No age crossed the 50% "
        "probability threshold."
    )


# ==========================================
# 8. FIND AGE WITH HIGHEST PROBABILITY
# ==========================================

highest_probability_row = (
    results_df
    .loc[
        results_df["probability"].idxmax()
    ]
)

highest_age = int(
    highest_probability_row["age"]
)

highest_probability = (
    highest_probability_row["probability"]
)


print("\n========================================")
print("HIGHEST PREDICTED PROBABILITY")
print("========================================")

print(
    f"Age: {highest_age}"
)

print(
    f"Probability: "
    f"{highest_probability * 100:.2f}%"
)


# ==========================================
# 9. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "results/metrics/age_probability.csv",
    index=False
)

print(
    "\nAge probability data saved to:"
    " results/metrics/age_probability.csv"
)


# ==========================================
# 10. PLOT
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["age"],
    results_df["probability"] * 100,
    marker="o"
)

plt.axhline(
    y=50,
    linestyle="--",
    label="50% threshold"
)

plt.xlabel("Age")
plt.ylabel("Perimenopause Probability (%)")

plt.title(
    "Age-wise Perimenopause Probability"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/figures/age_probability.png",
    dpi=300
)

plt.show()

print(
    "Graph saved to:"
    " results/figures/age_probability.png"
)
import joblib
import pandas as pd


# ==========================================
# 1. LOAD MODEL
# ==========================================

model = joblib.load(
    "models/menopause_classifier.pkl"
)


# ==========================================
# 2. USER INPUT
# ==========================================

age = 45
residence_type = 1
wealth_index = 3
is_pregnant = 0


# ==========================================
# 3. CURRENT PREDICTION
# ==========================================

user_data = pd.DataFrame([
    {
        "age": age,
        "residence_type": residence_type,
        "wealth_index": wealth_index,
        "is_pregnant": is_pregnant
    }
])


prediction = model.predict(
    user_data
)[0]

probability = model.predict_proba(
    user_data
)[0][1]


# ==========================================
# 4. PREDICTED STAGE
# ==========================================

if prediction == 1:

    predicted_stage = "Perimenopausal"

else:

    predicted_stage = (
        "Premenopausal / Perimenopausal"
    )


# ==========================================
# 5. AGE-WISE WINDOW ESTIMATION
# ==========================================

ages = list(range(30, 50))

age_probabilities = []


for test_age in ages:

    test_data = pd.DataFrame([
        {
            "age": test_age,
            "residence_type": residence_type,
            "wealth_index": wealth_index,
            "is_pregnant": is_pregnant
        }
    ])

    test_probability = model.predict_proba(
        test_data
    )[0][1]

    age_probabilities.append(
        {
            "age": test_age,
            "probability": test_probability
        }
    )


probability_df = pd.DataFrame(
    age_probabilities
)


# ==========================================
# 6. ESTIMATE WINDOW
# ==========================================

threshold = 0.50

window = probability_df[
    probability_df["probability"] >= threshold
]


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n========================================")
print("       MENOPAUSE AI PREDICTION")
print("========================================")

print(
    f"User age: {age}"
)

print(
    f"Predicted stage: {predicted_stage}"
)

print(
    f"Perimenopause probability: "
    f"{probability * 100:.2f}%"
)


if len(window) > 0:

    start_age = int(
        window["age"].iloc[0]
    )

    end_age = int(
        window["age"].iloc[-1]
    )

    print(
        f"Estimated transition window: "
        f"{start_age}-{end_age} years"
    )

else:

    print(
        "Estimated transition window: "
        "No age crossed the 50% threshold"
    )


print("========================================")
# Menocare — AI/ML

AI/ML module for the Menocare project, focused on menopause and perimenopause prediction using NFHS-based data.

## Current Work

- Data inspection and preprocessing
- Logistic Regression
- Random Forest
- XGBoost
- Model comparison
- Accuracy, Precision, Recall, F1 and ROC-AUC evaluation
- Confusion matrix and ROC curve
- Feature importance
- Trained model saving and prediction
- Age-wise probability and transition-window estimation

## Dataset

### NFHS Dataset

`data/processed/clean_nfhs_menopause.csv`

Features:
- `age`
- `residence_type`
- `wealth_index`
- `is_pregnant`

Target:
- `0` → Premenopausal / Perimenopausal
- `1` → Perimenopausal

### Symptom Dataset

`data/processed/clean_survey_symptoms.csv`

Reserved for the symptom prediction and tracking component.

## Project Structure

menocare/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_inspection.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── model_comparison.py
│   ├── evaluate_model.py
│   ├── feature_importance.py
│   ├── save_model.py
│   ├── estimate_window.py
│   └── predict.py
├── models/
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md

## Setup

Create a virtual environment and install the required dependencies:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Run

Run the ML scripts from the project root:

    python src/data_inspection.py
    python src/model_comparison.py
    python src/evaluate_model.py
    python src/feature_importance.py
    python src/save_model.py
    python src/predict.py

## Important Note

The current NFHS dataset does not contain an actual `age_at_menopause` target. Therefore, the age/window output is a model-derived estimate based on age-wise perimenopause probability, not an exact medical prediction.

## Team Integration

The trained ML model can later be integrated with the backend/API and frontend health application.
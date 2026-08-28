# Menocare — AI/ML

AI/ML module for Menocare, focused on estimating the menopause transition span across Indian regions using NFHS-based data.

## Objective

Estimate the menopause transition span for each region using NFHS data and machine learning.

The transition span is calculated from the difference between the median start age and median end age derived from menstrual-history data.

## Dataset

### NFHS Dataset

`data/raw/nfhs_menopause_subset.parquet`

Main variables used:

- `v012` — Age
- `v024` — Region
- `v025` — Residence type
- `v190` — Wealth index
- `v213` — Pregnancy status
- `v215` — Time since last menstrual period

### Generated Target

`data/processed/menopause_span_by_region.csv`

For each region:

- `Median_Start_Age`
- `Median_End_Age`
- `Menopause_Span`

Formula:

`Menopause_Span = Median_End_Age - Median_Start_Age`

## ML Approach

Region-level features are created from the NFHS data:

- Median age
- Mean age
- Urban rate
- Mean wealth index
- Pregnancy rate
- Sample size

Regression models evaluated:

- Linear Regression
- Random Forest
- XGBoost

XGBoost was selected based on the lowest MAE.

## Model Performance

Validation method: **Leave-One-Region-Out**

Regions evaluated: **36**

- MAE: **1.00 years**
- RMSE: **1.26 years**
- R²: **0.696**

The model's average prediction error is approximately 1 year in terms of menopause transition span.

## Prediction Example

For Region 5:

- Predicted transition span: **5.50 years**
- NFHS-derived median start age: **41 years**
- NFHS-derived median end age: **46 years**
- NFHS-derived transition span: **5 years**

## Project Structure

    menocare/
    ├── data/
    │   ├── raw/
    │   │   ├── nfhs_menopause_subset.parquet
    │   │   └── Menopause & Perimenopause Symptoms Survey.csv
    │   └── processed/
    │       ├── clean_nfhs_menopause.csv
    │       ├── clean_survey_symptoms.csv
    │       └── menopause_span_by_region.csv
    ├── models/
    │   └── menopause_span_model.pkl
    ├── results/
    │   ├── figures/
    │   │   └── menopause_span_loocv.png
    │   └── metrics/
    │       ├── menopause_span_models.csv
    │       ├── menopause_span_predictions.csv
    │       └── menopause_span_loocv.csv
    ├── src/
    │   ├── data_inspection.py
    │   ├── inspect_nfhs_raw.py
    │   ├── build_menopause_span.py
    │   ├── train_menopause_span.py
    │   ├── predict_menopause_span.py
    │   └── validate_menopause_span.py
    ├── notebooks/
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Setup

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Run

    python src/inspect_nfhs_raw.py
    python src/build_menopause_span.py
    python src/train_menopause_span.py
    python src/validate_menopause_span.py
    python src/predict_menopause_span.py

## Important Note

The menopause transition span is a regional research estimate derived from cross-sectional NFHS data. It is not an exact prediction of an individual's menopause age and should not be considered medical advice.

## Team Integration

The trained regression model can be integrated with the Menocare backend/API to provide region-level menopause transition span estimates to the application.
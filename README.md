# Menocare — Multimodal AI/ML System

An AI-based multimodal clinical decision support system for predicting perimenopause transition timelines, symptom burdens, and long-term health risks in Indian women.

The system is trained and validated on **strictly real Indian datasets**:
1. **NFHS-5 (National Family Health Survey, India)**: $N = 364,556$ women aged 30–49 across 36 Indian states and union territories.
2. **Menocare Clinical Symptom Survey**: $N = 28$ real respondents covering 21 perimenopausal/menopausal symptom domains.

---

## System Objectives & Modules

```text
                               ┌──────────────────────────────────────────────────────────┐
                               │                 Menocare AI Pipeline                     │
                               └──────────────────────────┬───────────────────────────────┘
                                                          │
         ┌────────────────────────┬───────────────────────┴───────────────┬────────────────────────┐
         │                        │                                       │                        │
         ▼                        ▼                                       ▼                        ▼
   [OBJECTIVE 1]            [OBJECTIVE 2]                           [OBJECTIVE 3]            [OBJECTIVE 4]
Symptom Prediction       Key Feature Importance                  Risk Stratification      Validation Suite
  & Onset Forecast         (Permutation & Odds Ratios)          (Low / Med / High)       (Stratified 5-Fold)
• Hazard & Survival      • Age, Region, Urban/Rural, Wealth      • Osteoporosis Risk      • Full NFHS-5 Cohort
• 12/24/36 mo windows    • Localized Indian Demographics         • Cardiovascular Risk    • Confusion Matrix & ROC
• Empirical Profiler                                             • Cognitive/Brain Fog    • Doctor Summary Report
```

### Objective 1: Transition Staging, Onset Forecasting & Symptom Profiler
* **Individual Stage Prediction**: Calibrated multi-class model classifying current status into *Premenopausal*, *Perimenopausal*, or *Postmenopausal* based on age, region, urban/rural residence, and wealth quintile.
* **Transition Onset Window Forecasting**: Computes cumulative hazard curves across age $30 \to 49$, forecasting the probability of transition onset within the next **12, 24, and 36 months** ahead.
* **Empirical Symptom Profiler**: Maps transition stage to empirical prevalence and severity scores (0 to 3) across 6 clinical clusters (Vasomotor, Psychological/Mood, Sleep, Somatic/Joints, Urogenital, Quality of Life).

### Objective 2: Mathematical Feature Attribution
* Uses mathematical analysis tools to quantify and rank risk drivers for Indian women:
  * **Permutation Feature Importance**: Evaluates drop in classification power upon feature perturbation.
  * **Tree Gini / Split Gain Importance**: Quantifies feature contribution in ensemble trees.
  * **Multinomial Logistic Regression Odds Ratios ($e^\beta$)**: Standardized odds with statistical significance.
* Key Findings: Biological **Age** is the predominant driver ($OR = 2.464$ for postmenopause), followed by **Geographic Region**, **Wealth Quintile** (nutritional/socioeconomic proxy), and **Residence Type** (urban lifestyle vs. rural physical activity).

### Objective 3: Triple-Condition Clinical Risk Stratification
Classifies women into **Low Risk / Medium Risk / High Risk** tiers for three major long-term health risks associated with estrogen withdrawal:
1. **Osteoporosis & Bone Mineral Density Loss**:
   * Adapted from the validated **OSTA (Osteoporosis Self-Assessment Tool for Asians)** and **SCORE** framework, incorporating estrogen depletion duration (`months_since_period`) and nutritional/wealth proxies.
   * Stratified into: *Low Risk* (routine monitoring), *Medium Risk* (osteopenia, Vitamin D3/Calcium intervention), *High Risk* (elevated fracture risk, DEXA scan indicated).
2. **Cardiovascular (CVD) 10-Year Event Risk**:
   * Adapted from the **WHO / International Society of Hypertension (ISH) South Asian Risk Charts** adjusted for menopausal vascular stiffening and urban lifestyle factors.
   * Stratified into: *Low Risk (<10%)*, *Medium Risk (10–20%)*, *High Risk (>20%)*.
3. **Cognitive Decline & Brain Fog Burden**:
   * Adapted from the **Menopause Rating Scale (MRS)** & **Greene Climacteric Psychological Subscale**, evaluating brain fog, memory lapses, sleep fragmentation, and emotional exhaustion.
   * Stratified into: *Low Risk* (mild/physiologic), *Medium Risk* (moderate disruption), *High Risk* (severe functional impairment).

### Objective 4: Clinical Validation on Indian Open Cohort (NFHS-5)
* Evaluated using **Stratified 5-Fold Cross-Validation** across all 364,556 NFHS-5 records:
  * Mean Out-of-Fold Balanced Accuracy: **0.534**
  * Mean Out-of-Fold Macro ROC-AUC: **0.727**
* Cross-validated regional generalizability with **Leave-One-Region-Out (LOOCV)** on the regional span model ($R^2 = 0.696$, MAE $= 1.00$ year across 36 Indian regions).
* Generates an automated **Clinical Assessment Report** designed for review by gynecologists and primary care physicians.

---

## Project Structure

```text
menocare/
├── data/
│   ├── raw/
│   │   ├── nfhs_menopause_subset.parquet          # 364,556 records (NFHS-5 demographic/menstrual)
│   │   └── Menopause & Perimenopause Symptoms Survey .csv  # 28 respondents (21 symptom domains)
│   └── processed/
│       ├── clean_nfhs_menopause.csv
│       ├── clean_survey_symptoms.csv
│       └── menopause_span_by_region.csv           # 36 Indian regions transition spans
│
├── models/
│   ├── menopause_span_model.pkl                   # [Preserved] Regional XGBoost span model
│   └── transition_stage_model.pkl                 # [New] Individual stage & hazard model
│
├── results/
│   ├── figures/
│   │   ├── confusion_matrix_stage.png             # 5-fold cross-validation confusion matrix
│   │   ├── feature_importance_attribution.png     # Feature importance attribution plots
│   │   ├── roc_curves_stage.png                   # Multi-class One-vs-Rest ROC curves
│   │   └── menopause_span_loocv.png               # Regional LOOCV scatter plot
│   └── metrics/
│       ├── feature_importance_analysis.csv        # Permutation, Gini, and Odds Ratios
│       ├── stage_model_performance.csv            # Test set classification metrics
│       ├── symptom_distribution_by_stage.csv      # Empirical prevalence & severity tables
│       ├── validation_report.csv                  # 5-fold cross-validation results
│       ├── menopause_span_loocv.csv
│       ├── menopause_span_models.csv
│       └── menopause_span_predictions.csv
│
├── src/
│   ├── symptom_prediction/
│   │   ├── stage_hazard_model.py                  # Objective 1: Staging & onset hazard forecasting
│   │   └── symptom_profiler.py                    # Objective 1: Empirical symptom severity mapping
│   ├── feature_importance/
│   │   └── feature_attribution.py                 # Objective 2: Mathematical feature ranking
│   ├── risk_stratification/
│   │   └── clinical_risk_engine.py                # Objective 3: Osteoporosis, CVD, Cognitive tiers
│   ├── validation/
│   │   └── cross_validation.py                    # Objective 4: Stratified 5-fold CV & plots
│   ├── pipeline.py                                # Unified multimodal patient CLI & test suite
│   ├── build_menopause_span.py                    # [Preserved] Regional span calculator
│   ├── train_menopause_span.py                    # [Preserved] Regional model trainer
│   ├── validate_menopause_span.py                 # [Preserved] Regional model LOOCV validator
│   └── predict_menopause_span.py                  # [Preserved] Regional model prediction CLI
│
├── notebooks/
│   └── MENOPAUSE.ipynb
├── requirements.txt
└── README.md
```

---

## Setup & Quick Start

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Unified Multimodal Test Suite
Run automated test cases across premenopausal, perimenopausal, and postmenopausal patient profiles:
```bash
python src/pipeline.py --test
```

### 3. Interactive Patient Clinical Assessment
Run the interactive decision-support CLI:
```bash
python src/pipeline.py
```

### 4. Run Individual Modules
* **Objective 1 (Transition Hazard & Onset Model)**:
  ```bash
  python src/symptom_prediction/stage_hazard_model.py
  python src/symptom_prediction/symptom_profiler.py
  ```
* **Objective 2 (Mathematical Feature Attribution)**:
  ```bash
  python src/feature_importance/feature_attribution.py
  ```
* **Objective 3 (Clinical Risk Stratification Engine)**:
  ```bash
  python src/risk_stratification/clinical_risk_engine.py
  ```
* **Objective 4 (5-Fold Cross-Validation)**:
  ```bash
  python src/validation/cross_validation.py
  ```
* **Regional Menopause Span Model (Original)**:
  ```bash
  python src/predict_menopause_span.py
  ```

---

## Important Medical Note

This software is an epidemiological and clinical decision support tool designed for research and screening guidance. It computes statistical probabilities and clinical risk tiers based on validated Indian population distributions. It is not an automated medical diagnosis and must be interpreted in consultation with qualified healthcare professionals.
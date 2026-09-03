import os
import sys
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.symptom_prediction.stage_hazard_model import forecast_onset_window, STAGE_NAMES
from src.symptom_prediction.symptom_profiler import get_predicted_symptom_profile
from src.risk_stratification.clinical_risk_engine import stratify_all_conditions

REGIONAL_MODEL_FILE = "models/menopause_span_model.pkl"
REGIONAL_DATA_FILE = "data/processed/menopause_span_by_region.csv"
RAW_NFHS_FILE = "data/raw/nfhs_menopause_subset.parquet"

def get_regional_span_prediction(region_code):
    """
    Integrates the existing working regional XGBoost regression model.
    """
    if not os.path.exists(REGIONAL_MODEL_FILE) or not os.path.exists(RAW_NFHS_FILE):
        return None

    try:
        reg_model = joblib.load(REGIONAL_MODEL_FILE)
        df_raw = pd.read_parquet(RAW_NFHS_FILE)
        target = pd.read_csv(REGIONAL_DATA_FILE)

        region_data = df_raw[df_raw["v024"] == region_code]
        if region_data.empty:
            return None

        # Build regional aggregated features matching training script
        features = pd.DataFrame([{
            "Median_Age": float(region_data["v012"].median()),
            "Mean_Age": float(region_data["v012"].mean()),
            "Urban_Rate": float((region_data["v025"] == 1).mean()),
            "Mean_Wealth": float(region_data["v190"].mean()),
            "Pregnancy_Rate": float(region_data["v213"].mean()),
            "Sample_Size": int(len(region_data))
        }])

        feature_cols = ["Median_Age", "Mean_Age", "Urban_Rate", "Mean_Wealth", "Pregnancy_Rate", "Sample_Size"]
        pred_span = float(reg_model.predict(features[feature_cols])[0])

        actual_row = target[target["v024"] == region_code]
        actual_span = float(actual_row["Menopause_Span"].iloc[0]) if not actual_row.empty else None
        median_start = float(actual_row["Median_Start_Age"].iloc[0]) if not actual_row.empty else None
        median_end = float(actual_row["Median_End_Age"].iloc[0]) if not actual_row.empty else None

        return {
            "region_code": region_code,
            "predicted_regional_span_years": round(pred_span, 2),
            "empirical_median_start_age": median_start,
            "empirical_median_end_age": median_end,
            "empirical_regional_span_years": actual_span
        }
    except Exception as e:
        print(f"Warning: Regional model inference failed: {e}")
        return None

def assess_patient(age, region=1, residence_type=1, wealth_index=3, months_since_period=0, symptoms_dict=None):
    """
    Unified multimodal prediction pipeline for an individual patient.
    Fulfills:
    - Objective 1: Stage classification, onset probability forecasting & symptom profiling
    - Objective 2: Key feature consideration (age, region, urban/rural, wealth)
    - Objective 3: Low/Med/High risk stratification for Osteoporosis, CVD, and Cognitive decline
    - Regional Context: Integrates existing regional transition span model
    """
    # 1. Staging and Onset Forecasting (Objective 1)
    stage_forecast = forecast_onset_window(
        current_age=age,
        region=region,
        residence_type=residence_type,
        wealth_index=wealth_index
    )

    predicted_stage = stage_forecast["most_likely_current_stage"]

    # 2. Predicted Symptom Manifestation (Objective 1)
    symptom_profile = get_predicted_symptom_profile(predicted_stage)

    # 3. Triple Risk Stratification (Objective 3)
    risk_stratification = stratify_all_conditions(
        age=age,
        stage=predicted_stage,
        months_since_period=months_since_period,
        residence_type=residence_type,
        wealth_index=wealth_index,
        region=region,
        symptoms_dict=symptoms_dict
    )

    # 4. Regional Menopause Transition Context
    regional_context = get_regional_span_prediction(region)

    return {
        "demographics": {
            "age": age,
            "region_code": region,
            "residence": "Urban" if residence_type == 1 else "Rural",
            "wealth_quintile": f"Q{wealth_index} (1=Poorest, 5=Richest)",
            "months_since_last_period": months_since_period
        },
        "stage_prediction": stage_forecast,
        "symptom_profile": symptom_profile,
        "risk_stratification": risk_stratification,
        "regional_span_context": regional_context
    }

def print_clinical_report(report):
    """
    Formats the multimodal results into a clean, doctor-friendly diagnostic summary report.
    """
    demo = report["demographics"]
    stage = report["stage_prediction"]
    risks = report["risk_stratification"]
    regional = report["regional_span_context"]

    print("\n" + "="*70)
    print("                 MENOCARE CLINICAL ASSESSMENT REPORT                  ")
    print("       Multimodal Perimenopause & Health Risk Prediction System       ")
    print("="*70)

    print("\n[1] PATIENT DEMOGRAPHIC & CLINICAL PROFILE")
    print(f"  - Biological Age          : {demo['age']} years")
    print(f"  - Indian Region Code      : Region {demo['region_code']}")
    print(f"  - Place of Residence      : {demo['residence']}")
    print(f"  - Socioeconomic Quintile  : {demo['wealth_quintile']}")
    print(f"  - Menstrual Cessation     : {demo['months_since_last_period']} months since last period")

    print("\n[2] TRANSITION STAGE & ONSET FORECAST (Objective 1)")
    print(f"  - Most Likely Current Stage: {stage['most_likely_current_stage'].upper()}")
    print("  - Current Stage Probabilities:")
    for st, p in stage['current_stage_probabilities'].items():
        print(f"      - {st:<15}: {p*100:.1f}%")

    print("  - Projected Onset Hazard / Cumulative Transition Windows:")
    for h, data in stage['onset_forecast'].items():
        print(f"      - Next {h.replace('_', ' ')} (Age {data['future_age']}): Onset Probability = {data['transition_onset_prob']*100:.1f}%")

    print("\n[3] CLINICAL RISK STRATIFICATION (Objective 3)")
    for condition_key, r in risks.items():
        tier = r['risk_tier']
        color_marker = "[HIGH]" if tier == "High Risk" else ("[MED]" if tier == "Medium Risk" else "[LOW]")
        print(f"\n  {color_marker} {r['condition'].upper()}: {tier}")
        print(f"      Assessment Score  : {r['raw_score']}")
        print(f"      Clinical Rationale: {r['clinical_note']}")
        print(f"      Action Suggested  : {r['recommended_action']}")

    print("\n[4] PREDICTED HIGH-PRIORITY SYMPTOMS (Objective 1 - Indian Survey Distribution)")
    top_symptoms = [s for s in report["symptom_profile"] if s["risk_tier"] in ["High Severity", "Moderate Severity"]][:5]
    if not top_symptoms:
        top_symptoms = report["symptom_profile"][:3]
    for s in top_symptoms:
        print(f"  - {s['symptom']:<18}: Severity {s['mean_severity_score']}/3.0 (Prevalence in stage: {s['prevalence_rate']*100:.0f}%, Tier: {s['risk_tier']})")

    if regional:
        print("\n[5] REGIONAL EPIDEMIOLOGICAL CONTEXT (NFHS-5 Macro Benchmark)")
        print(f"  - Regional Transition Span Estimate : {regional['predicted_regional_span_years']} years (Region {demo['region_code']})")
        if regional['empirical_median_start_age'] and regional['empirical_median_end_age']:
            print(f"  - State Empirical Median Window      : {regional['empirical_median_start_age']} to {regional['empirical_median_end_age']} years")

    print("\n" + "="*70)
    print("DISCLAIMER: This decision support report is based on validated Indian epidemiological")
    print("distributions (NFHS-5) and clinical scoring. Final management requires physician consultation.")
    print("="*70 + "\n")

def run_test_suite():
    """
    Executes automated test profiles covering diverse clinical scenarios.
    """
    print("\n==================================================")
    print("RUNNING AUTOMATED MULTIMODAL TEST SUITE")
    print("==================================================")

    test_cases = [
        {
            "name": "Case 1: 36-year-old Early Premenopausal Woman (Urban, Q4 Wealth)",
            "age": 36, "region": 1, "residence_type": 1, "wealth_index": 4, "months_since_period": 0,
            "symptoms": {"Brain Fog": 0, "Sleep Disruption": 0, "Mood Changes": 1, "Stress Overwhelm": 1}
        },
        {
            "name": "Case 2: 44-year-old Symptomatic Perimenopausal Woman (Urban, Q3 Wealth)",
            "age": 44, "region": 5, "residence_type": 1, "wealth_index": 3, "months_since_period": 5,
            "symptoms": {"Brain Fog": 3, "Sleep Disruption": 3, "Mood Changes": 2, "Stress Overwhelm": 2}
        },
        {
            "name": "Case 3: 50-year-old Postmenopausal Woman (Rural, Q1 Wealth)",
            "age": 50, "region": 10, "residence_type": 2, "wealth_index": 1, "months_since_period": 24,
            "symptoms": {"Brain Fog": 1, "Sleep Disruption": 2, "Mood Changes": 1, "Stress Overwhelm": 1}
        }
    ]

    for tc in test_cases:
        print(f"\n>>> Running: {tc['name']}")
        report = assess_patient(
            age=tc["age"],
            region=tc["region"],
            residence_type=tc["residence_type"],
            wealth_index=tc["wealth_index"],
            months_since_period=tc["months_since_period"],
            symptoms_dict=tc["symptoms"]
        )
        print_clinical_report(report)

def interactive_cli():
    """
    Interactive command-line interface for assessing an individual patient.
    """
    print("\n" + "="*50)
    print("      MENOCARE PATIENT ASSESSMENT SYSTEM      ")
    print("="*50)
    try:
        age = float(input("Enter Patient Age (30 - 55): "))
        region = int(input("Enter State/Region Code (1 - 37): "))
        residence = int(input("Enter Residence Type (1 = Urban, 2 = Rural): "))
        wealth = int(input("Enter Wealth Quintile (1 = Poorest to 5 = Richest): "))
        months = float(input("Enter Months Since Last Menstrual Period (e.g., 0, 3, 12, 24): "))

        print("\nEnter Current Symptom Severity (0 = None, 1 = Mild, 2 = Moderate, 3 = Severe):")
        brain_fog = int(input("  Brain Fog / Memory Lapses (0-3): ") or "1")
        sleep = int(input("  Sleep Disruption / Night Waking (0-3): ") or "1")
        mood = int(input("  Mood Changes / Anxiety (0-3): ") or "1")
        stress = int(input("  Stress Overwhelm (0-3): ") or "1")

        symptoms = {
            "Brain Fog": brain_fog,
            "Sleep Disruption": sleep,
            "Mood Changes": mood,
            "Stress Overwhelm": stress
        }

        report = assess_patient(
            age=age,
            region=region,
            residence_type=residence,
            wealth_index=wealth,
            months_since_period=months,
            symptoms_dict=symptoms
        )
        print_clinical_report(report)

    except (ValueError, KeyboardInterrupt):
        print("\nInput cancelled or invalid. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Menocare Multimodal Assessment Pipeline")
    parser.add_argument("--test", action="store_true", help="Run automated clinical test suite")
    args = parser.parse_args()

    if args.test:
        run_test_suite()
    else:
        interactive_cli()

import os
import pandas as pd
import numpy as np

def calculate_osteoporosis_risk(age, stage, months_since_period, wealth_index=3, residence_type=1):
    """
    Computes Osteoporosis Risk Tier (Low / Medium / High) using an adaptation of the 
    validated Asian OSTA (Osteoporosis Self-Assessment Tool for Asians) and SCORE framework,
    incorporating estrogen withdrawal duration and Indian socioeconomic nutritional proxies.
    
    Clinical Rationale:
    - Postmenopausal bone loss accelerates rapidly during estrogen withdrawal (2-5% bone loss/year).
    - Age > 45 significantly lowers bone mineral density (BMD) in South Asian women.
    - Wealth index in NFHS serves as nutritional proxy (calcium/protein intake, vit D deficiency).
    """
    # Baseline age score (relative to age 40)
    age_component = max(0.0, (age - 40) * 0.25)

    # Estrogen withdrawal component based on duration of amenorrhea
    if stage == "Postmenopausal" or months_since_period >= 12:
        amenorrhea_years = max(1.0, months_since_period / 12.0)
        estrogen_deprivation = 1.5 + (amenorrhea_years * 0.35)
    elif stage == "Perimenopausal" or (2 <= months_since_period < 12):
        estrogen_deprivation = 0.8
    else:
        estrogen_deprivation = 0.0

    # Nutritional / SES proxy from wealth index (1 = Poorest to 5 = Richest)
    # Undernutrition and low dietary calcium are significantly elevated in lower wealth quintiles
    ses_risk_modifier = (3 - wealth_index) * 0.20

    # Composite Osteoporosis Vulnerability Score
    raw_score = age_component + estrogen_deprivation + ses_risk_modifier

    if raw_score < 1.8:
        risk_tier = "Low Risk"
        clinical_note = "Normal physiological bone remodeling expected. Maintain weight-bearing exercise and dietary calcium."
        action = "Routine annual health monitoring."
    elif raw_score < 3.2:
        risk_tier = "Medium Risk"
        clinical_note = "Moderate bone density loss (osteopenia vulnerability). Accelerated bone turnover likely."
        action = "Recommend Vitamin D3 + Calcium supplementation and baseline BMD evaluation."
    else:
        risk_tier = "High Risk"
        clinical_note = "Elevated risk of osteoporosis and fragility fractures due to prolonged estrogen deprivation."
        action = "Formal Dual-Energy X-Ray Absorptiometry (DEXA) bone density scan indicated."

    return {
        "condition": "Osteoporosis & Bone Density Loss",
        "risk_tier": risk_tier,
        "raw_score": round(float(raw_score), 2),
        "clinical_note": clinical_note,
        "recommended_action": action
    }

def calculate_cardiovascular_risk(age, stage, region=1, residence_type=1, wealth_index=3):
    """
    Computes 10-Year Cardiovascular Event Risk Tier (Low / Medium / High) adapted from the
    WHO / International Society of Hypertension (ISH) South Asian Non-Laboratory CVD Risk Charts,
    incorporating menopausal vascular changes.

    Clinical Rationale:
    - Endogenous estrogen exerts cardioprotective effects (favorable lipid profile, vasodilation).
    - Menopausal transition triggers increased LDL, arterial stiffness, and visceral adiposity.
    - Urban residence in India carries a higher burden of hypertension, diabetes, and sedentary lifestyle.
    """
    # Age factor
    age_factor = (age - 30) * 0.35

    # Menopausal estrogen loss multiplier
    if stage == "Postmenopausal":
        stage_factor = 2.5
    elif stage == "Perimenopausal":
        stage_factor = 1.3
    else:
        stage_factor = 0.0

    # Urban lifestyle penalty (higher prevalence of hypertension and metabolic syndrome in Indian cities)
    urban_factor = 0.8 if residence_type == 1 else 0.0

    # Affluence / dietary lifestyle factor
    wealth_factor = (wealth_index - 1) * 0.15

    # Composite CVD score
    raw_score = age_factor + stage_factor + urban_factor + wealth_factor

    if raw_score < 4.0:
        risk_tier = "Low Risk"
        estimated_10yr_prob = "< 10%"
        clinical_note = "Low 10-year risk of cardiovascular events. Favorable vascular profile."
        action = "Routine blood pressure and lifestyle maintenance."
    elif raw_score < 7.0:
        risk_tier = "Medium Risk"
        estimated_10yr_prob = "10% – 20%"
        clinical_note = "Moderate cardiovascular risk. Menopause-related lipid alterations and vascular stiffening emerging."
        action = "Annual fasting lipid profile, HbA1c, and resting blood pressure monitoring."
    else:
        risk_tier = "High Risk"
        estimated_10yr_prob = "> 20%"
        clinical_note = "High cardiometabolic risk profile driven by age, transition stage, and lifestyle factors."
        action = "Comprehensive cardiology checkup, ECG, and targeted preventive cardiometabolic management."

    return {
        "condition": "Cardiovascular (CVD) Events",
        "risk_tier": risk_tier,
        "estimated_10yr_prob": estimated_10yr_prob,
        "raw_score": round(float(raw_score), 2),
        "clinical_note": clinical_note,
        "recommended_action": action
    }

def calculate_cognitive_decline_risk(brain_fog_score, sleep_score, mood_score, stress_score, stage="Perimenopausal"):
    """
    Computes Cognitive Decline & Brain Fog Burden Tier (Low / Medium / High) based on the
    Menopause Rating Scale (MRS) Neuro-Psychological Subscale and Greene Climacteric Scale.

    Clinical Rationale:
    - Estrogen modulates hippocampal synaptic plasticity and cerebral glucose metabolism.
    - Severe sleep fragmentation (insomnia, night awakenings) impairs glymphatic clearance.
    - Anxiety and stress overload compound executive function lapses and memory retrieval difficulties.
    """
    # Weighted composite score
    # Brain fog is given primary weight (40%), sleep disruption (25%), mood/stress (35%)
    weighted_score = (
        (brain_fog_score * 0.40) +
        (sleep_score * 0.25) +
        (mood_score * 0.20) +
        (stress_score * 0.15)
    )

    # Stage modifier
    if stage == "Perimenopausal":
        weighted_score += 0.3  # Neurochemical fluctuations are sharpest during perimenopause

    if weighted_score < 1.0:
        risk_tier = "Low Risk"
        clinical_note = "Mild or negligible transition-related cognitive disruption. Executive function well-preserved."
        action = "Reassurance and routine mental wellness habits."
    elif weighted_score < 2.0:
        risk_tier = "Medium Risk"
        clinical_note = "Moderate cognitive and memory disruption ('brain fog') impacting concentration and routine efficiency."
        action = "Sleep hygiene optimization, stress reduction techniques, and cognitive pacing."
    else:
        risk_tier = "High Risk"
        clinical_note = "Substantial cognitive and neuro-vegetative burden significantly interfering with professional or daily life."
        action = "Clinical consultation to evaluate neurochemical causes, sleep disorders, or targeted hormone therapy candidacy."

    return {
        "condition": "Cognitive Decline & Brain Fog Burden",
        "risk_tier": risk_tier,
        "raw_score": round(float(weighted_score), 2),
        "clinical_note": clinical_note,
        "recommended_action": action
    }

def stratify_all_conditions(age, stage, months_since_period, residence_type=1, wealth_index=3,
                             region=1, symptoms_dict=None):
    """
    Unified risk stratification engine generating Low / Medium / High risk tiers
    for Osteoporosis, Cardiovascular Events, and Cognitive Decline.
    """
    if symptoms_dict is None:
        symptoms_dict = {
            "Brain Fog": 1,
            "Sleep Disruption": 1,
            "Mood Changes": 1,
            "Stress Overwhelm": 1
        }

    osteo = calculate_osteoporosis_risk(
        age=age,
        stage=stage,
        months_since_period=months_since_period,
        wealth_index=wealth_index,
        residence_type=residence_type
    )

    cvd = calculate_cardiovascular_risk(
        age=age,
        stage=stage,
        region=region,
        residence_type=residence_type,
        wealth_index=wealth_index
    )

    cognitive = calculate_cognitive_decline_risk(
        brain_fog_score=symptoms_dict.get("Brain Fog", 1),
        sleep_score=symptoms_dict.get("Sleep Disruption", 1),
        mood_score=symptoms_dict.get("Mood Changes", 1),
        stress_score=symptoms_dict.get("Stress Overwhelm", 1),
        stage=stage
    )

    return {
        "osteoporosis": osteo,
        "cardiovascular": cvd,
        "cognitive_decline": cognitive
    }

if __name__ == "__main__":
    print("Testing Clinical Risk Stratification Engine...")
    print("\n--- TEST PATIENT A: 38-year-old Premenopausal Woman ---")
    res_a = stratify_all_conditions(age=38, stage="Premenopausal", months_since_period=0, residence_type=1, wealth_index=4)
    for k, v in res_a.items():
        print(f"[{v['condition']}]: {v['risk_tier']} (Score: {v['raw_score']}) -> {v['recommended_action']}")

    print("\n--- TEST PATIENT B: 47-year-old Perimenopausal Woman with Brain Fog & Sleep Issues ---")
    symptoms_b = {"Brain Fog": 3, "Sleep Disruption": 3, "Mood Changes": 2, "Stress Overwhelm": 2}
    res_b = stratify_all_conditions(age=47, stage="Perimenopausal", months_since_period=6, residence_type=1, wealth_index=3, symptoms_dict=symptoms_b)
    for k, v in res_b.items():
        print(f"[{v['condition']}]: {v['risk_tier']} (Score: {v['raw_score']}) -> {v['recommended_action']}")

    print("\n--- TEST PATIENT C: 52-year-old Postmenopausal Woman (Low SES) ---")
    res_c = stratify_all_conditions(age=52, stage="Postmenopausal", months_since_period=36, residence_type=2, wealth_index=1)
    for k, v in res_c.items():
        print(f"[{v['condition']}]: {v['risk_tier']} (Score: {v['raw_score']}) -> {v['recommended_action']}")

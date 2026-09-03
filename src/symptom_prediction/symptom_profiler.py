import os
import pandas as pd
import numpy as np

SURVEY_FILE = "data/raw/Menopause & Perimenopause Symptoms Survey .csv"
SUMMARY_METRICS = "results/metrics/symptom_distribution_by_stage.csv"

# Standard numeric severity mapping for survey response scales (0 = none, 1 = mild/rare, 2 = moderate, 3 = severe/frequent)
SCALE_MAPPING = {
    # Hot flashes
    "Never": 0,
    "Rarely (1–2 times a month)": 1,
    "Occasionally (1–2 times a week)": 2,
    "Frequently (Daily, 1–3 times a day)": 3,
    
    # Night sweats
    "Rarely (A few times a month)": 1,
    "Almost every night": 3,
    
    # Sleep disruption
    "Excellent (Sleep through the night easily)": 0,
    "Mildly disrupted (Occasional trouble falling or staying asleep)": 1,
    "Moderately disrupted (Frequent waking, night sweats, or insomnia)": 2,
    "Severely disrupted (Chronic insomnia, waking multiple times a night)": 3,
    
    # Mood changes
    "No noticeable changes": 0,
    "Mild (Slight mood swings, easily manageable)": 1,
    "Moderate (Noticeable anxiety, frequent irritability, or uncharacteristic mood dips)": 2,
    "Severe (Significantly impacts personal or professional life)": 3,
    
    # Brain fog
    "Rarely": 1,
    "Sometimes (Occasional forgetfulness)": 2,
    "Frequently (Difficulty with word recall, concentration, or mental clarity daily)": 3,
    
    # Physical fatigue
    "Never / Rarely": 0,
    "Mild energy dips occasionally": 1,
    "Moderate fatigue most days": 2,
    "Severe exhaustion impacting daily function": 3,
    
    # Joint pain
    "Occasionally": 2,
    "Frequently / Daily": 3,
    
    # Intimate physical changes
    "No symptoms": 0,
    "Mild discomfort occasionally": 1,
    "Moderate symptoms affecting comfort/intimacy": 2,
    "Severe discomfort requiring medical attention or daily management": 3,
    
    # Stress tolerance
    "No change": 0,
    "Mildly lower tolerance": 1,
    "Moderately lower tolerance": 2,
    "Frequently overwhelmed": 3,
    
    # Heart palpitations
    "Very rarely": 1,
    "Occasionally": 2,
    "Frequently (Multiple times a week or during episodes of anxiety)": 3,
    
    # Digestive / Bloating
    "Almost daily": 3,
    
    # Overall Quality of Life Impact
    "No impact": 0,
    "Mild impact (Noticeable, but easily managed)": 1,
    "Moderate impact (Requires lifestyle adjustments or support)": 2,
    "Severe impact (Significantly impairs daily activities or mental health)": 3
}

SYMPTOM_COLUMNS = {
    "Hot Flashes": "How often do you experience hot flashes or sudden flushes of heat?",
    "Night Sweats": "How frequently do night sweats disrupt your sleep?",
    "Sleep Disruption": "How would you rate the quality of your sleep over the past 3–6 months?",
    "Mood Changes": "Have you noticed changes in your mood (e.g., irritability, sudden anxiety, low mood)?",
    "Brain Fog": 'Do you experience "brain fog" (memory lapses, poor concentration, difficulty finding words)?',
    "Fatigue": "How often do you experience unexplained physical fatigue or low energy levels?",
    "Joint Pain": "Do you experience joint pain, muscle stiffness, or body aches not related to exercise or injury?",
    "Intimate Changes": "Have you experienced intimate physical changes (e.g., vaginal dryness, discomfort, or urinary urgency)?  ",
    "Stress Overwhelm": "Have you noticed changes in your tolerance to stress or sudden feelings of being overwhelmed by routine tasks?  ",
    "Palpitations": "Have you experienced sudden heart palpitations or a racing heartbeat without physical exertion?",
    "Digestive Issues": "How frequently do you experience sudden bloating or digestive issues (gas, acid reflux, new food sensitivities)?",
    "QoL Impact": "Overall, how severely do these symptoms impact your day-to-day quality of life?"
}

def classify_survey_menstrual_stage(status_string):
    """
    Classifies survey response into Premenopausal, Perimenopausal, or Postmenopausal.
    """
    status_lower = str(status_string).lower()
    if "12 consecutive months" in status_lower or "surgical" in status_lower:
        return "Postmenopausal"
    elif "irregular" in status_lower or "less than 12" in status_lower:
        return "Perimenopausal"
    return "Premenopausal"

def load_and_parse_survey(filepath=SURVEY_FILE):
    """
    Loads raw survey and converts ordinal symptom questions into numerical severity scales.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Survey file not found at: {filepath}")

    df = pd.read_csv(filepath)
    parsed = pd.DataFrame()

    # Stage
    stage_col = "Which statement best describes your current menstrual status?"
    parsed["Menopause_Stage"] = df[stage_col].apply(classify_survey_menstrual_stage)
    
    # Age Range
    age_col = "What is your current age range?"
    parsed["Age_Range"] = df[age_col]

    # Map symptoms to numerical severity 0-3
    for name, col_name in SYMPTOM_COLUMNS.items():
        if col_name in df.columns:
            parsed[name] = df[col_name].map(SCALE_MAPPING).fillna(0).astype(int)

    return parsed

def compute_empirical_symptom_distributions():
    """
    Computes empirical prevalence (severity >= 1) and average severity (0-3)
    across stages directly from real survey data.
    """
    os.makedirs("results/metrics", exist_ok=True)
    df = load_and_parse_survey()

    stages = ["Premenopausal", "Perimenopausal", "Postmenopausal"]
    records = []

    symptom_names = list(SYMPTOM_COLUMNS.keys())

    for stage in stages:
        stage_df = df[df["Menopause_Stage"] == stage]
        n_stage = len(stage_df)
        if n_stage == 0:
            continue

        for sym in symptom_names:
            prevalence = (stage_df[sym] >= 1).mean() * 100
            mean_severity = stage_df[sym].mean()
            records.append({
                "Menopause_Stage": stage,
                "Sample_Size": n_stage,
                "Symptom": sym,
                "Prevalence_Pct": round(prevalence, 1),
                "Mean_Severity_0to3": round(mean_severity, 2)
            })

    dist_df = pd.DataFrame(records)
    dist_df.to_csv(SUMMARY_METRICS, index=False)
    print(f"Empirical symptom distributions saved to: {SUMMARY_METRICS}")
    return dist_df

def get_predicted_symptom_profile(predicted_stage):
    """
    Given a predicted transition stage (e.g. from the NFHS hazard model),
    returns empirical symptom risk and rank-ordered manifestation profile.
    """
    df = load_and_parse_survey()
    stage_df = df[df["Menopause_Stage"] == predicted_stage]

    # Fallback to perimenopausal if stage has few samples
    if len(stage_df) < 3:
        stage_df = df[df["Menopause_Stage"] == "Perimenopausal"]

    profile = []
    for sym in SYMPTOM_COLUMNS.keys():
        prev = (stage_df[sym] >= 1).mean()
        mean_sev = stage_df[sym].mean()
        high_sev_pct = (stage_df[sym] >= 2).mean()

        # Risk level determination
        if mean_sev >= 1.5 or high_sev_pct >= 0.5:
            risk_tier = "High Severity"
        elif mean_sev >= 0.7 or prev >= 0.5:
            risk_tier = "Moderate Severity"
        else:
            risk_tier = "Mild / Low Severity"

        profile.append({
            "symptom": sym,
            "prevalence_rate": round(float(prev), 3),
            "mean_severity_score": round(float(mean_sev), 2),
            "high_severity_rate": round(float(high_sev_pct), 3),
            "risk_tier": risk_tier
        })

    # Sort symptoms by severity and prevalence
    profile.sort(key=lambda x: (x["mean_severity_score"], x["prevalence_rate"]), reverse=True)
    return profile

if __name__ == "__main__":
    print("Parsing Menopause & Perimenopause Symptoms Survey...")
    summary = compute_empirical_symptom_distributions()
    print(summary.head(15))

    print("\nSample Symptom Profile for Predicted Perimenopausal Stage:")
    perimenopause_profile = get_predicted_symptom_profile("Perimenopausal")
    for s in perimenopause_profile[:6]:
        print(f"  - {s['symptom']:<18}: Severity = {s['mean_severity_score']}/3.0 (Prevalence: {s['prevalence_rate']*100:.0f}%, Tier: {s['risk_tier']})")

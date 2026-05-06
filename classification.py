import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
import streamlit as st
import boto3
from botocore.exceptions import ClientError
import datetime

# Streamlit dashboard
st.set_page_config(page_title="ENCHANTED Model 1", layout="wide")

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background: linear-gradient(135deg, #f7fbff 0%, #eef6fb 45%, #f8fafc 100%);
        color: #1f2937;
    }

    /* Main content container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header title */
    h1 {
        color: #0b3a66;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    h2, h3 {
        color: #1f4e79;
        font-weight: 700;
    }

    /* Subtle dashboard cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #dbeafe;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        background: white;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        padding: 8px;
    }

    /* Text area */
    textarea {
        background-color: #f8fafc !important;
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        color: #1f2937 !important;
    }

    /* Select box and input styling */
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #cbd5e1;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #0b5cab, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.22);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #084b8a, #1d4ed8);
        color: white;
        border: none;
    }

    /* Info box */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Horizontal divider */
    hr {
        border: none;
        height: 1px;
        background: #dbeafe;
        margin: 1.5rem 0;
    }

    /* Make dataframe list/pill tag text bold */
    [data-testid="stDataFrame"] div[data-baseweb="tag"],
    [data-testid="stDataFrame"] div[data-baseweb="tag"] span,
    [data-testid="stDataFrame"] span[data-baseweb="tag"] {
        font-weight: 700 !important;
        color: #000000 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


features = [
    "age", "los_days", "days_to_edd", "copd_flag",
    "systolic_bp", "diastolic_bp", "heart_rate", "temperature",
    "spo2", "oxygen_flow_rate", "news2",
    "hb", "platelet", "anc", "sodium", "potassium",
    "pending_surgery_flag", "active_procedure_flag",
    "active_precaution_flag", "active_iv_med_flag"
]


def rule_based_screening(row):
    red_flags = []
    amber_flags = []

    if row["pregnancy_flag"] == 1:
        red_flags.append("Pregnancy")

    if row["news2"] >= 5:
        red_flags.append("NEWS2 >= 5")
    elif row["news2"] > 0:
        amber_flags.append("NEWS2 > 0")

    if row["oxygen_flow_rate"] > 2:
        red_flags.append("Oxygen flow > 2 L/min")
    elif row["oxygen_flow_rate"] > 0:
        amber_flags.append("Low-flow O2")

    if row["copd_flag"] == 1:
        if row["spo2"] < 88:
            red_flags.append("COPD SpO2 < 88")
        elif row["spo2"] <= 92:
            amber_flags.append("COPD SpO2 88–92")
    else:
        if row["spo2"] < 91:
            red_flags.append("SpO2 < 91")
        elif row["spo2"] < 96:
            amber_flags.append("SpO2 < 96")

    if row["temperature"] >= 38:
        red_flags.append("Temperature >= 38")
    elif row["temperature"] >= 37.5:
        amber_flags.append("Temperature 37.5–37.9")

    if row["heart_rate"] >= 120:
        red_flags.append("Heart rate >= 120")
    elif row["heart_rate"] >= 100:
        amber_flags.append("Heart rate 100–119")

    if row["pending_surgery_flag"] == 1:
        red_flags.append("Pending surgery")

    if row["active_iv_med_flag"] == 1:
        red_flags.append("IV medication")

    if row["active_procedure_flag"] == 1:
        amber_flags.append("Procedure order")

    if row["active_precaution_flag"] == 1:
        amber_flags.append("Precaution order")

    if red_flags:
        category = "Red - No-Go"
    elif amber_flags:
        category = "Amber - Review Required"
    else:
        category = "Green - Potential Candidate"

    return category, red_flags, amber_flags


def risk_band(prob):
    if prob >= 0.30:
        return "High Risk"
    elif prob >= 0.15:
        return "Medium Risk"
    else:
        return "Low Risk"


def generate_basic_explanation(row):
    reasons = []

    if row["news2"] > 0:
        reasons.append(f"NEWS2 score is {row['news2']}")

    if row["oxygen_flow_rate"] > 0:
        reasons.append(f"Patient is on oxygen flow {row['oxygen_flow_rate']} L/min")

    if row["spo2"] < 96:
        reasons.append(f"SpO2 is {row['spo2']}%")

    if row["pending_surgery_flag"] == 1:
        reasons.append("Patient has pending surgery")

    if not reasons:
        return "Patient appears clinically stable based on available screening parameters."

    return "Key factors: " + "; ".join(reasons)


def build_llm_prompt(row):
    """
    Builds a safe prompt for the LLM explanation layer.
    The LLM should only summarise provided structured data.
    """

    prompt = f"""
Summary of patient's clinical screening and AI risk assessment:

Patient ID: {row["patient_id"]}
Encounter ID: {row["encounter_id"]}

Patient screening output:
- Rule-based category: {row["rule_category"]}
- Red flags: {row["red_flags"]}
- Amber flags: {row["amber_flags"]}
- Predictive risk score: {row["risk_score"]}
- Predictive risk band: {row["risk_band"]}

Key clinical values:
- Age: {row["age"]}
- COPD flag: {row["copd_flag"]}
- Systolic BP: {row["systolic_bp"]}
- Diastolic BP: {row["diastolic_bp"]}
- Heart rate: {row["heart_rate"]}
- Temperature: {row["temperature"]}
- SpO2: {row["spo2"]}
- Oxygen device: {row["oxygen_device"]}
- Oxygen flow rate: {row["oxygen_flow_rate"]}
- NEWS2: {row["news2"]}
- Hb: {row["hb"]}
- Platelet: {row["platelet"]}
- ANC: {row["anc"]}
- Sodium: {row["sodium"]}
- Potassium: {row["potassium"]}
- Pending surgery flag: {row["pending_surgery_flag"]}
- Active procedure flag: {row["active_procedure_flag"]}
- Active precaution flag: {row["active_precaution_flag"]}
- Active IV medication flag: {row["active_iv_med_flag"]}

AI-supported recommendation: {row["ai_recommendation"]}

Please produce the response in this exact format:

**1. Explanation of Screening Output**
Provide a short explanation of why the patient received this screening output.

**2. Key Review Points**
List key points that the case manager or clinician should review.

**3. Suggested Review Status**
State whether the patient is:
- Not recommended for CH referral screening at this stage based on rule-based red flag exclusion; or
- Shortlisted for case manager review; or
- Potential Community Hospital candidate for case manager review.

Do not make a final transfer decision.

**4. Clinical Decision Reminder**
State that this is AI-supported decision support only and that the final referral / transfer decision remains with the case manager, clinician, and care team.
"""
    return prompt

@st.cache_resource
def get_bedrock_client():
    session = boto3.Session(
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )

    return session.client(
        service_name="bedrock-runtime",
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )

def call_bedrock_llm(prompt: str) -> str:
    client = get_bedrock_client()

    model_id = "anthropic.claude-3-haiku-20240307-v1:0"  # change based on your approved Bedrock model

    try:
        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            inferenceConfig={
                "maxTokens": 500,
                "temperature": 0.2
            }
        )

        return response["output"]["message"]["content"][0]["text"]

    except ClientError as e:
        return f"Bedrock ClientError: {e.response['Error']['Message']}"

    except Exception as e:
        return f"Unexpected error calling Bedrock: {str(e)}"


def generate_llm_explanation_placeholder(row):
    """
    Placeholder for future LLM explanation.
    For now, this returns the prompt instead of calling an actual LLM.
    """

    return build_llm_prompt(row)


# Load data
shortlisted = pd.read_csv("shortlisted.csv")

# Temporary synthetic outcome label for prototype testing only
# Replace this with real outcome data later
shortlisted["rebound_72h"] = (
    (shortlisted["news2"] >= 5) |
    (shortlisted["oxygen_flow_rate"] > 2) |
    (shortlisted["spo2"] < 91) |
    (shortlisted["hb"] < 9) |
    (shortlisted["sodium"] < 130) |
    (shortlisted["potassium"] > 5.5)
).astype(int)

shortlisted[["rule_category", "red_flags", "amber_flags"]] = shortlisted.apply(
    lambda row: pd.Series(rule_based_screening(row)),
    axis=1
)

MODEL_PATH = "models/enchanted_model1_random_forest.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Initialise risk outputs
shortlisted["risk_score"] = None
shortlisted["risk_band"] = "Not applicable"

# Only apply AI model to Amber / Green patients
eligible_mask = shortlisted["rule_category"].isin([
    "Amber - Review Required",
    "Green - Potential Candidate"
])

# Generate AI risk scores only for eligible patients
if eligible_mask.any():
    shortlisted.loc[eligible_mask, "risk_score"] = model.predict_proba(
        shortlisted.loc[eligible_mask, features]
    )[:, 1]

    shortlisted.loc[eligible_mask, "risk_band"] = shortlisted.loc[
        eligible_mask, "risk_score"
    ].apply(risk_band)


def ai_review_recommendation(row):
    if row["rule_category"] == "Red - No-Go":
        return "Not recommended based on rule-based red flag exclusion"

    if row["risk_band"] == "High Risk":
        return "Shortlisted but requires priority clinical review"

    if row["risk_band"] == "Medium Risk":
        return "Shortlisted for case manager review"

    if row["risk_band"] == "Low Risk":
        return "Potential CH candidate for case manager review"

    return "Pending review"

shortlisted["ai_recommendation"] = shortlisted.apply(ai_review_recommendation, axis=1)

# Generate explanation / LLM prompt
shortlisted["llm_prompt"] = shortlisted.apply(generate_llm_explanation_placeholder, axis=1)


st.title("ENCHANTED Model 1: CH Referral Screening Dashboard")
st.subheader("Rule-Based Screening and AI-Supported Risk Stratification")

st.caption(
    "For demonstration using sample data. Final referral and transfer decisions remain with the clinical care team."
)

st.markdown(
    """
    <div style="
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #dbeafe;
        border-left: 6px solid #2563eb;
        padding: 18px 22px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        margin-bottom: 20px;
    ">
        <div style="font-size: 18px; font-weight: 700; color: #0b3a66;">
            Acute-to-Community Hospital Referral Decision Support
        </div>
        <div style="font-size: 14px; color: #475569; margin-top: 6px;">
            This dashboard combines rule-based clinical screening with AI-supported risk stratification to assist
            case managers and clinicians in reviewing potential Community Hospital referral candidates.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap;">
        <span style="background:#fee2e2; color:#7f1d1d; padding:8px 14px; border-radius:999px; font-weight:600;">
            Red: Rule-Based Exclusion
        </span>
        <span style="background:#fef3c7; color:#78350f; padding:8px 14px; border-radius:999px; font-weight:600;">
            Amber: Clinical Review Required
        </span>
        <span style="background:#dcfce7; color:#14532d; padding:8px 14px; border-radius:999px; font-weight:600;">
            Green: Potential Candidate
        </span>
        <span style="background:#dbeafe; color:#1e3a8a; padding:8px 14px; border-radius:999px; font-weight:600;">
            AI: Risk Stratification
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Summary metrics
total_patients = len(shortlisted)
red_count = (shortlisted["rule_category"] == "Red - No-Go").sum()
amber_count = (shortlisted["rule_category"] == "Amber - Review Required").sum()
green_count = (shortlisted["rule_category"] == "Green - Potential Candidate").sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", total_patients)
col2.metric("Red / No-Go", red_count)
col3.metric("Amber / Review", amber_count)
col4.metric("Green / Candidate", green_count)

st.divider()

# Display patient table
st.subheader("Patient Screening Results")

def format_flags(flags):
    if isinstance(flags, list) and len(flags) > 0:
        return ", ".join(flags)
    return "-"

display_cols = [
    "patient_id",
    "encounter_id",
    "rule_category",
    "red_flags",
    "amber_flags",
    "risk_score",
    "risk_band",
    "ai_recommendation",
    # "llm_prompt"
]

# st.dataframe(shortlisted[display_cols], use_container_width=True)

def colour_rule_category(value):
    if value == "Green - Potential Candidate":
        return "background-color: #d4edda; color: #155724;"
    if value == "Potential CH candidate for case manager review":
        return "background-color: #d4edda; color: #155724;"
    if value == "Shortlisted for case manager review":
        return "background-color: #fff3cd; color: #856404;"
    if value == "Shortlisted but requires priority clinical review":
        return "background-color: #fff3cd; color: #856404;"
    if value == "Not recommended based on rule-based red flag exclusion":
        return "background-color: #f8d7da; color: #721c24;"
    if value == "Amber - Review Required":
        return "background-color: #fff3cd; color: #856404;"
    if value == "Red - No-Go":
        return "background-color: #f8d7da; color: #721c24;"
    return ""

styled_df = shortlisted[display_cols].style.map(
    colour_rule_category,
    subset=["rule_category", "ai_recommendation"]
)

st.dataframe(
    styled_df,
    use_container_width=True,
    column_config={
        "patient_id": st.column_config.TextColumn("Patient ID", width="medium"),
        "encounter_id": st.column_config.TextColumn("Encounter ID", width="medium"),
        "rule_category": st.column_config.TextColumn("Rule Category", width="large"),
        "red_flags": st.column_config.ListColumn("Red Flags", width="large"),
        "amber_flags": st.column_config.ListColumn("Amber Flags", width="large"),
        "risk_score": st.column_config.NumberColumn("Risk Score", width="small", format="%.2f"),
        "risk_band": st.column_config.TextColumn("Risk Band", width="medium"),
        "ai_recommendation": st.column_config.TextColumn("AI Recommendation", width="large"),
        "llm_prompt": None
    }
)

st.divider()

# Patient-level view
st.subheader("Patient Detail View")

selected_patient = st.selectbox(
    "Select patient",
    shortlisted["patient_id"].tolist()
)

patient_row = shortlisted[shortlisted["patient_id"] == selected_patient].iloc[0]

st.write("### Screening Output")
st.write(f"**Rule-based category:** {patient_row['rule_category']}")
st.write(f"**Red flags:** {patient_row['red_flags']}")
st.write(f"**Amber flags:** {patient_row['amber_flags']}")
# safer version
if pd.notna(patient_row["risk_score"]):
    st.write(f"**Predictive risk score:** {patient_row['risk_score']:.2f}")
else:
    st.write("**Predictive risk score:** Not applicable")
st.write(f"**Predictive risk band:** {patient_row['risk_band']}")

st.write(f"**AI-supported recommendation:** *{patient_row['ai_recommendation']}*")

st.write("### Case Manager / Clinician Review")

final_decision = st.selectbox(
    "Final referral decision",
    [
        "Pending Review",
        "Suitable for CH Referral",
        "Not Suitable for CH Referral",
        "Requires Further Clinical Clarification"
    ]
)

review_comments = st.text_area("Review comments / override reason")

st.info(
    "The AI model provides decision support only. "
    "Final referral decisions remain with the case manager / clinical team."
)

if st.button("Submit Review Decision"):
    review_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": patient_row["patient_id"],
        "encounter_id": patient_row["encounter_id"],
        "rule_category": patient_row["rule_category"],
        "risk_score": patient_row["risk_score"],
        "risk_band": patient_row["risk_band"],
        "ai_recommendation": patient_row["ai_recommendation"],
        "final_decision": final_decision,
        "review_comments": review_comments
    }

    review_log_path = "case_manager_review_log.csv"

    try:
        existing_log = pd.read_csv(review_log_path)
        updated_log = pd.concat(
            [existing_log, pd.DataFrame([review_record])],
            ignore_index=True
        )
    except FileNotFoundError:
        updated_log = pd.DataFrame([review_record])

    updated_log.to_csv(review_log_path, index=False)

    st.success("Review decision submitted and saved to audit log.")

if st.checkbox("Show submitted review log"):
    try:
        review_log = pd.read_csv("case_manager_review_log.csv")
        st.dataframe(review_log, use_container_width=True)
    except FileNotFoundError:
        st.info("No review decisions have been submitted yet.")

st.write("### LLM Prompt")
st.text_area(
    "Prompt that would be sent to the LLM explanation layer",
    patient_row["llm_prompt"],
    height=400
)

st.write("### LLM-Generated Explanation")

if st.button("Generate LLM Explanation"):
    with st.spinner("Generating explanation..."):
        llm_output = call_bedrock_llm(patient_row["llm_prompt"])

    st.markdown(llm_output)

import os
import ast
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

load_dotenv()


@st.cache_resource(show_spinner=False)
def load_granite_model():
    api_key = os.getenv("api_key")
    region = os.getenv("region")
    project_id = os.getenv("project_id")
    creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
    return ModelInference(model_id="ibm/granite-3-3-8b-instruct", credentials=creds, project_id=project_id)


model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


symptoms = [
    "fever", "cough", "headache", "fatigue",
    "shortness_of_breath", "chest_pain", "nausea", "sore_throat"
]

for key in ["clear_input", "awaiting_ack", "predicted_result", "uncheck_checkboxes"]:
    if key not in st.session_state:
        st.session_state[key] = False


st.sidebar.header("🧝 Patient Details")
patient_name = st.sidebar.text_input("Name")
patient_age = st.sidebar.text_input("Age")
patient_gender = st.sidebar.selectbox("Gender", ["", "Male", "Female", "Other"])
medical_history = st.sidebar.text_area("Medical History")
current_medications = st.sidebar.text_area("Current Medications")
allergies = st.sidebar.text_area("Allergies")

st.sidebar.subheader("📊 Vitals & Recent Info")
avg_heart_rate = st.sidebar.text_input("Average Heart Rate (bpm)", value="80")
avg_bp_systolic = st.sidebar.text_input("Avg. BP Systolic", value="120")
avg_bp_diastolic = st.sidebar.text_input("Avg. BP Diastolic", value="80")
avg_glucose = st.sidebar.text_input("Average Glucose (mg/dL)", value="100")
recent_symptoms = st.sidebar.text_area("Recently Reported Symptoms")


st.set_page_config(page_title="LLM Disease Predictor", page_icon="🧠")
st.title("🧠 LLM-powered Disease Predictor")
st.markdown("Enter your symptoms in **natural language**, or 📋select below:")


if st.session_state.clear_input:
    st.session_state.clear_input = False
    st.session_state.main_input = ""
    st.session_state.predicted_result = None
    st.rerun()


user_text = st.text_input("🔤 What's bothering you today?", key="main_input")


checkbox_input = []
cols = st.columns(3)
for i, symptom in enumerate(symptoms):
    key = f"symptom_{symptom}"
    if st.session_state.uncheck_checkboxes:
        st.session_state[key] = False
    with cols[i % 3]:
        selected = st.checkbox(symptom.capitalize(), key=key)
        checkbox_input.append(1 if selected else 0)
if st.session_state.uncheck_checkboxes:
    st.session_state.uncheck_checkboxes = False


def extract_symptoms_from_text(text):
    prompt = f"""
You are a medical assistant. Extract only the known symptoms from the following patient message.

Known symptoms: {', '.join(symptoms)}

Patient says: \"{text}\"

Return ONLY a valid Python list of matching symptoms like [\"fever\", \"nausea\"]. Do not include any explanation or extra text.
"""
    model = load_granite_model()
    try:
        response = ""
        for _ in range(10):
            chunk = model.generate_text(prompt)
            if not chunk.strip():
                break
            response += chunk
            prompt += chunk
        extracted = ast.literal_eval(response.strip())
        if not isinstance(extracted, list):
            extracted = []
        return [1 if symptom in extracted else 0 for symptom in symptoms], extracted
    except Exception:
        return [0] * len(symptoms), []


if st.button("🔍 Predict"):
    llm_features, extracted_list = extract_symptoms_from_text(user_text)
    final_features = [max(c, l) for c, l in zip(checkbox_input, llm_features)]

    if sum(final_features) == 0:
        st.session_state.clarify_needed = True
        st.session_state.original_input = user_text
        st.session_state.extracted_text = extracted_list
        st.session_state.predicted_result = None
    else:
        st.session_state.clarify_needed = False
        input_df = pd.DataFrame([final_features], columns=symptoms)
        prediction = model.predict(input_df)[0]
        disease = label_encoder.inverse_transform([prediction])[0]

        prediction_prompt = f"""
As a medical AI assistant, predict potential health conditions based on the following patient information.

Current Symptoms: {', '.join([s for s, v in zip(symptoms, final_features) if v == 1])}
Age: {patient_age}
Gender: {patient_gender}
Medical History: {medical_history}
Current Medications: {current_medications}
Allergies: {allergies}

Recent Health Metrics:
Average Heart Rate: {avg_heart_rate} bpm
Average Blood Pressure: {avg_bp_systolic}/{avg_bp_diastolic} mmHg
Average Blood Glucose: {avg_glucose} mg/dL
Recently Reported Symptoms: {recent_symptoms}

Format your response as:
1. Potential condition name
2. Likelihood (High/Medium/Low)
3. Brief explanation
4. Recommended next steps

Provide the top 3 most likely conditions based on the data provided.
"""
        granite_model = load_granite_model()
        llm_response = ""
        try:
            for _ in range(10):
                chunk = granite_model.generate_text(prediction_prompt)
                if not chunk.strip():
                    break
                llm_response += chunk
                prediction_prompt += chunk
        except Exception as e:
            llm_response = f"⚠️ LLM Prediction Error: {e}"

        result = {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "symptoms": [s for s, v in zip(symptoms, final_features) if v == 1] + [s for s in extracted_list if s not in symptoms],
            "prediction": disease,
            "llm_analysis": llm_response.strip()
        }

        st.session_state.predicted_result = result
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(result)
        st.session_state.uncheck_checkboxes = True
        st.rerun()


if st.session_state.predicted_result:
    res = st.session_state.predicted_result
    st.success(f"🧲 {res['name']}, based on your symptoms, the predicted disease (ML) is: **{res['prediction']}**")
    st.markdown(f"• Age: {res['age']}")
    st.markdown(f"• Gender: {res['gender']}")
    st.markdown(f"• Symptoms: `{', '.join(res['symptoms'])}`")
    st.markdown("### 🧠 LLM-Based Prediction")
    st.markdown(res['llm_analysis'])
    st.markdown("---")

st.caption("⚠️ This tool is not a substitute for professional medical advice.")

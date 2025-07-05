import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import joblib
import os


model = joblib.load("risk_model.joblib")
label_map = joblib.load("label_map.joblib")


CSV_FILE = "patient_health_data.csv"
SYMPTOMS = ["None", "Headache", "Nausea", "Fatigue", "Dizziness", "Chest Pain"]


if "patient_data" not in st.session_state:
    if os.path.exists(CSV_FILE):
        st.session_state.patient_data = pd.read_csv(CSV_FILE).to_dict(orient="records")
    else:
        st.session_state.patient_data = []


st.set_page_config("📊 Patient Dashboard", layout="wide")
st.title("🧠 HealthAI - Intelligent Healthcare Assistant")
st.subheader("📋 Enter Patient Data Below")


with st.form("health_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        entry_date = st.date_input("Date", value=date.today())
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 74)
        glucose = st.number_input("Blood Glucose (mg/dL)", 50, 250, 101)
    
    with col2:
        systolic = st.number_input("Systolic BP", 90, 180, 120)
        diastolic = st.number_input("Diastolic BP", 60, 120, 80)
        sleep = st.number_input("Sleep Hours", 0.0, 12.0, 6.8)
    
    with col3:
        symptom = st.selectbox("Symptoms", SYMPTOMS)

    submitted = st.form_submit_button("➕ Add Entry")

    if submitted:
        
        symptom_code = SYMPTOMS.index(symptom)
        features = [[heart_rate, glucose, systolic, diastolic, sleep, symptom_code]]
        prediction = model.predict(features)[0]
        risk_label = label_map[prediction]

        
        new_record = {
            "Date": str(entry_date),
            "Heart Rate": heart_rate,
            "Systolic BP": systolic,
            "Diastolic BP": diastolic,
            "Blood Glucose": glucose,
            "Sleep": sleep,
            "Symptoms": symptom,
            "Predicted Risk": risk_label
        }

        st.session_state.patient_data.append(new_record)

        
        pd.DataFrame(st.session_state.patient_data).to_csv(CSV_FILE, index=False)

        st.success(f"✅ Entry added! Predicted Health Risk: **{risk_label}**")


if st.session_state.patient_data:
    df = pd.DataFrame(st.session_state.patient_data)
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', errors='coerce').dt.date

    df.sort_values("Date", inplace=True)

    st.markdown("---")
    st.subheader("📈 Health Analytics Dashboard")

    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(df, x="Date", y="Heart Rate", title="Heart Rate Trend"), use_container_width=True)
    with col2:
        st.plotly_chart(px.line(df, x="Date", y=["Systolic BP", "Diastolic BP"], title="Blood Pressure Trend"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        glucose_fig = px.line(df, x="Date", y="Blood Glucose", title="Blood Glucose Trend")
        glucose_fig.add_hline(y=140, line_dash="dash", line_color="red", annotation_text="High Glucose")
        st.plotly_chart(glucose_fig, use_container_width=True)

    with col4:
        symptom_counts = df["Symptoms"].value_counts()
        st.plotly_chart(px.pie(names=symptom_counts.index, values=symptom_counts.values, title="Symptom Frequency"), use_container_width=True)

    
    st.subheader("📊 Metrics Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. Heart Rate", f"{df['Heart Rate'].mean():.1f} bpm")
    col2.metric("Avg. BP", f"{df['Systolic BP'].mean():.0f}/{df['Diastolic BP'].mean():.0f}")
    col3.metric("Avg. Glucose", f"{df['Blood Glucose'].mean():.1f} mg/dL")
    col4.metric("Avg. Sleep", f"{df['Sleep'].mean():.1f} hrs")

    
    st.subheader("📌 Patient Data with Predictions")
    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Patient Data as CSV", csv_data, file_name="patient_health_data.csv", mime="text/csv")
else:
    st.info("ℹ️ No data yet. Please add at least one record.")

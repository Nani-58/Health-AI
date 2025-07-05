import os
from dotenv import load_dotenv
import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from fpdf import FPDF
from pathlib import Path


load_dotenv()


@st.cache_resource(show_spinner=False)
def init_granite_model():
    project_id = os.getenv("project_id")
    api_key = os.getenv("api_key")
    region = os.getenv("region")

    creds = Credentials(
        api_key=api_key,
        url=f"https://{region}.ml.cloud.ibm.com"
    )

    return ModelInference(
        model_id="ibm/granite-3-3-8b-instruct",
        credentials=creds,
        project_id=project_id
    )
def create_pdf(name, age, gender, condition, medical_history, current_medications, allergies, treatment_plan):
    pdf = FPDF()
    pdf.add_page()

    font_path = "DejaVuSans.ttf"
    if Path(font_path).exists():
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
        text = (
            f"Patient Name: {name}\nAge: {age}\nGender: {gender}\n"
            f"Condition: {condition}\nMedical History: {medical_history}\n"
            f"Current Medications: {current_medications}\nAllergies: {allergies}\n\n"
            f"Treatment Plan:\n{treatment_plan}"
        )
    else:
        pdf.set_font("Arial", size=12)
        clean_plan = treatment_plan.encode("ascii", "ignore").decode()
        text = (
            f"Patient Name: {name}\nAge: {age}\nGender: {gender}\n"
            f"Condition: {condition}\nMedical History: {medical_history}\n"
            f"Current Medications: {current_medications}\nAllergies: {allergies}\n\n"
            f"Treatment Plan:\n{clean_plan}"
        )

    pdf.multi_cell(0, 10, text)
    file_path = "treatment_plan_card_style.pdf"
    pdf.output(file_path)
    return file_path


def generate_treatment(condition, age, gender, medical_history, current_medications, allergies):
    model = init_granite_model()
    prompt = f"""
You are a professional and empathetic medical AI assistant.

Patient details:
- Condition: {condition}
- Age: {age}
- Gender: {gender}
- Medical History: {medical_history}
- Current Medications: {current_medications}
- Allergies: {allergies}

Give a short 3-4 line summary about the condition's causes and typical symptoms.
Then provide a detailed treatment plan with at least 5 clearly numbered sections:
1. Recommended medications with dosage (consider allergies)
2. Lifestyle/habit changes
3. Required follow-up or diagnostic tests
4. Diet changes (foods to prefer/avoid)
5. Physical & mental wellness tips

Avoid jargon. Make it understandable for a regular person.
"""
    full_response = ""
    try:
        for _ in range(13):  
            chunk = model.generate_text(prompt + full_response)
            if not chunk.strip():
                break
            full_response += chunk.strip() + "\n"
    except Exception as e:
        return f"⚠️ Error generating treatment: {str(e)}"

    return full_response.strip()


st.set_page_config("HealthAI Treatment Generator", page_icon="💊")
st.title("🧠 HealthAI - Intelligent Healthcare Assistant")
st.sidebar.header("Patient Profile")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
medical_history = st.sidebar.text_area("Medical History", "None")
current_medications = st.sidebar.text_area("Current Medications", "None")
allergies = st.sidebar.text_area("Allergies", "None")


condition = st.text_input("Enter the Medical Condition")


if "treatment_history" not in st.session_state:
    st.session_state.treatment_history = []

if st.button("💊 Generate Treatment Plan"):
    if condition.strip():
        with st.spinner("Generating personalized plan..."):
            treatment_plan = generate_treatment(condition, age, gender, medical_history, current_medications, allergies)

            
            st.session_state.treatment_history.append({
                "name": name,
                "age": age,
                "gender": gender,
                "condition": condition,
                "medical_history": medical_history,
                "current_medications": current_medications,
                "allergies": allergies,
                "treatment": treatment_plan
            })

        
            file_path = create_pdf(name, age, gender, condition, medical_history, current_medications, allergies, treatment_plan)


            st.subheader("📋 Personalized Treatment Plan")
            st.markdown(f"""
            <div style='padding:20px;border-radius:10px;background-color:#f9f9f9;border:1px solid #ddd;'>
                <p><b>Condition:</b> {condition}</p>
                <p><b>Age:</b> {age}</p>
                <p><b>Gender:</b> {gender}</p>
                <p><b>Medical History:</b> {medical_history}</p>
                <p><b>Current Medications:</b> {current_medications}</p>
                <p><b>Allergies:</b> {allergies}</p>
                <hr/>
                <pre>{treatment_plan}</pre>
            </div>
            """, unsafe_allow_html=True)

        
            with open(file_path, "rb") as f:
                st.download_button("📄 Download as PDF", f, file_name="treatment_plan.pdf")
    else:
        st.warning("⚠️ Please enter a medical condition to generate a plan.")


if st.session_state.treatment_history:
    st.subheader("🕘 Treatment History")
    for record in st.session_state.treatment_history:
        with st.expander(f"{record['name']} ({record['age']}) - {record['condition']}"):
            st.markdown(f"""
            **Gender:** {record['gender']}  
            **Medical History:** {record['medical_history']}  
            **Current Medications:** {record['current_medications']}  
            **Allergies:** {record['allergies']}  
            ---
            ```
{record['treatment']}
            ```
            """)

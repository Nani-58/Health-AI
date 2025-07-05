import os
from dotenv import load_dotenv
import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

# Load environment variables
load_dotenv()

@st.cache_resource(show_spinner=False)
def init_granite_model():
    project_id = os.getenv("project_id")
    api_key = os.getenv("api_key")
    region = os.getenv("region")
    model_id = "ibm/granite-3-3-8b-instruct"

    creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
    model = ModelInference(model_id=model_id, credentials=creds, project_id=project_id)
    return model

# Sidebar patient information
with st.sidebar:
    st.header("🧑‍⚕️ Patient Details")
    st.session_state.patient_name = st.text_input("Name", value=st.session_state.get("patient_name", ""))
    st.session_state.patient_age = st.text_input("Age", value=st.session_state.get("patient_age", ""))
    st.session_state.patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
    st.session_state.medical_history = st.text_area("Medical History", height=80)
    st.session_state.current_medications = st.text_area("Current Medications", height=80)
    st.session_state.allergies = st.text_area("Allergies", height=80)

# 🧠 Generate AI response with rich prompt
def generate_response(query):
    model = init_granite_model()
    short_history = st.session_state.chat_history[-10:]

    patient_context = f"""
You are a helpful healthcare AI assistant.

Patient Profile:
Name: {st.session_state.patient_name}
Age: {st.session_state.patient_age}
Gender: {st.session_state.patient_gender}
Medical History: {st.session_state.medical_history or 'N/A'}
Current Medications: {st.session_state.current_medications or 'N/A'}
Allergies: {st.session_state.allergies or 'N/A'}

Instructions:
As a healthcare AI assistant, provide a helpful, accurate, and evidence-based response to the patient's question below.

Make sure your response:
- Directly addresses the question.
- Includes relevant medical facts.
- Acknowledges limitations when appropriate.
- Suggests when to seek professional medical advice.
- Avoids making definitive diagnoses.
- Uses accessible, non-technical language.
"""

    # Add recent chat for context
    prompt = patient_context + "\n\nRecent Conversation:\n"
    for sender, message in short_history:
        prompt += f"{sender.upper()}: {message}\n"

    prompt += f"\nPATIENT QUESTION:\n{query}\n\nRESPONSE:\n"

    # Generate response in chunks
    response = ""
    for _ in range(10):
        try:
            chunk = model.generate_text(prompt)
        except Exception as e:
            st.error(f"⚠️ Error generating response: {e}")
            break
        if not chunk.strip():
            break
        response += chunk
        prompt += chunk
    return response

# 🔷 App layout
st.set_page_config(page_title="HealthAI Chatbot", page_icon="💬")
st.title("🩺 HealthAI Chatbot")
st.markdown("Chat with the intelligent healthcare assistant below.")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "run_example" not in st.session_state:
    st.session_state.run_example = False

# Send message
def send_message():
    user_input = st.session_state.user_input.strip()
    if user_input:
        with st.spinner("Generating response..."):
            ai_response = generate_response(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", ai_response))
        st.session_state.user_input = ""

# Run example input
if st.session_state.run_example:
    send_message()
    st.session_state.run_example = False

# 💬 Display chat history
for idx, (sender, message) in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if sender == "You" else "ai"):
        st.markdown(message)
        if sender == "AI":
            st.radio("Was this helpful?", ["👍 Yes", "👎 No"], key=f"feedback_{idx}", horizontal=True)

# 🛠️ Control buttons and example queries
with st.container():
    col1, col2, col3, col4 = st.columns([1.2, 2, 2, 2])
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.user_input = ""
    examples = ["What are symptoms of diabetes?", "How to reduce fever?", "How to control migraine?"]
    for i in range(3):
        with [col2, col3, col4][i]:
            if st.button(examples[i]):
                st.session_state.user_input = examples[i]
                st.session_state.run_example = True
                st.rerun()

# 🧾 Input bar at bottom
st.text_input("Ask a health-related question:", key="user_input", on_change=send_message)

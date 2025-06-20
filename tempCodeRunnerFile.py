# Final model without function calling using llms
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

import streamlit as st

# Set up environment variables
model = os.environ.get("MODEL", "llama3.1")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 8))

# Initialize the embeddings and database
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

# Initialize the LLM
llm = Ollama(model=model)

# Define prompt template
prompt_template = """You are a medical assistant providing information based on verified medical documents. 
Use the following pieces of context to answer the question at the end. If you don't know the answer, say that you don't know, don't try to make up an answer.
Always provide a detailed response of at least 30-40 words, focusing on accurate medical information.

Context: {context}

Question: {question}

Detailed Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Initialize the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Database setup
DB_NAME = "appointments.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def create_tables():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         patient_name TEXT,
         date TEXT,
         time TEXT,
         hospital TEXT,
         appointment_type TEXT)
        ''')
        
        # Check if appointment_type column exists, if not, add it
        cursor.execute("PRAGMA table_info(appointments)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'appointment_type' not in columns:
            cursor.execute('ALTER TABLE appointments ADD COLUMN appointment_type TEXT')
        
        conn.commit()

create_tables()

# Function definitions
def is_valid_appointment_time(date_str, time_str, appointment_type):
    appointment_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    current_datetime = datetime.now()
    
    if appointment_datetime <= current_datetime:
        return False, "Cannot book appointments in the past."
    
    if appointment_type == "general":
        hour = appointment_datetime.hour
        if not ((9 <= hour < 12) or (17 <= hour < 23)):
            return False, "General appointments can only be booked between 9:00 AM to 12:00 PM and 5:00 PM to 11:00 PM."
    
    return True, ""

def check_conflicting_appointment(date, time, hospital):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        SELECT * FROM appointments 
        WHERE date = ? AND time = ? AND hospital = ?
        ''', (date, time, hospital))
        existing_appointment = cursor.fetchone()
    
    return existing_appointment is not None

def suggest_alternative_time(date, time, hospital):
    appointment_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    alternative_datetime = appointment_datetime + timedelta(minutes=30)
    return alternative_datetime.strftime("%Y-%m-%d"), alternative_datetime.strftime("%H:%M")

def book_appointment(patient_name: str, date: str, time: str, hospital: str, appointment_type: str) -> str:
    is_valid, error_message = is_valid_appointment_time(date, time, appointment_type)
    if not is_valid:
        return error_message

    if check_conflicting_appointment(date, time, hospital):
        alt_date, alt_time = suggest_alternative_time(date, time, hospital)
        return f"The requested time slot is already booked. Would you like to book for {alt_date} at {alt_time} instead?"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO appointments (patient_name, date, time, hospital, appointment_type)
        VALUES (?, ?, ?, ?, ?)
        ''', (patient_name, date, time, hospital, appointment_type))
        conn.commit()
    return f"Appointment booked successfully for {patient_name} at {hospital} on {date} at {time} for {appointment_type} appointment."

def view_appointments() -> List[Dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM appointments')
        appointments = cursor.fetchall()
    
    return [
        {
            "id": app[0],
            "patient_name": app[1],
            "date": app[2],
            "time": app[3],
            "hospital": app[4],
            "appointment_type": app[5]
        }
        for app in appointments
    ]

def medical_qa(query: str) -> str:
    result = qa.invoke({"query": query})
    return result['result']

# Streamlit UI
st.set_page_config(page_title="Medical Assistant", layout="wide")
st.title("Medical Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_step" not in st.session_state:
    st.session_state.current_step = "initial"
if "appointment_info" not in st.session_state:
    st.session_state.appointment_info = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if st.session_state.current_step == "initial" and "book an appointment" in prompt.lower():
            full_response = "I'd be happy to help you book an appointment. Is this a general appointment or an emergency appointment?"
            st.session_state.current_step = "appointment_type"
        elif st.session_state.current_step == "appointment_type":
            if prompt.lower() in ["general", "emergency"]:
                st.session_state.appointment_info["appointment_type"] = prompt.lower()
                full_response = "Could you please provide the patient's name?"
                st.session_state.current_step = "name"
            else:
                full_response = "Please specify if this is a 'general' or 'emergency' appointment."
        elif st.session_state.current_step == "name":
            st.session_state.appointment_info["patient_name"] = prompt
            full_response = f"Thank you, {prompt}. What date would you like to book the appointment for? (Please use the format YYYY-MM-DD)"
            st.session_state.current_step = "date"
        elif st.session_state.current_step == "date":
            st.session_state.appointment_info["date"] = prompt
            full_response = "Great. What time would you prefer for the appointment? (Please use the 24-hour format HH:MM)"
            st.session_state.current_step = "time"
        elif st.session_state.current_step == "time":
            is_valid, error_message = is_valid_appointment_time(
                st.session_state.appointment_info["date"],
                prompt,
                st.session_state.appointment_info["appointment_type"]
            )
            if not is_valid:
                full_response = f"{error_message} Please provide a new time for the appointment (24-hour format HH:MM):"
            else:
                st.session_state.appointment_info["time"] = prompt
                full_response = "Noted. Which hospital would you like to book the appointment at?"
                st.session_state.current_step = "hospital"
        elif st.session_state.current_step == "hospital":
            st.session_state.appointment_info["hospital"] = prompt
            appointment_response = book_appointment(**st.session_state.appointment_info)
            full_response = f"{appointment_response}\nIs there anything else I can help you with?"
            st.session_state.current_step = "initial"
            st.session_state.appointment_info = {}
        elif "view appointments" in prompt.lower():
            appointments = view_appointments()
            if appointments:
                full_response = "Here are the current appointments:\n"
                for app in appointments:
                    full_response += f"- {app['patient_name']} at {app['hospital']} on {app['date']} at {app['time']} ({app['appointment_type']} appointment)\n"
            else:
                full_response = "There are currently no appointments scheduled."
        else:
            full_response = medical_qa(prompt)

        for chunk in full_response.split():
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar
with st.sidebar:
    st.title("Chat Information")
    st.write(f"Model: {model}")
    st.write(f"Embeddings: {embeddings_model_name}")
    st.write(f"Target chunks: {target_source_chunks}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.current_step = "initial"
        st.session_state.appointment_info = {}
        st.experimental_rerun()
    
    st.title("View Appointments")
    if st.button("Show Appointments"):
        appointments = view_appointments()
        st.json(appointments)

if __name__ == "__main__":
    pass

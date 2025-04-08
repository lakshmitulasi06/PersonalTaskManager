import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import time
import json
import os
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from plyer import notification
from google.oauth2 import service_account
from googleapiclient.discovery import build
import speech_recognition as sr
import pywhatkit as kt
import pyautogui
import webbrowser

# Set up Streamlit Page
st.set_page_config(page_title="Enhanced Task Manager ✅", page_icon="✅", layout="wide")

# Initialize session state
if "tasks" not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=["Task", "Details", "Date", "Time", "Reminder", "VoiceNote"])
if "messages" not in st.session_state:
    st.session_state.messages = []
if "notepad" not in st.session_state:
    st.session_state.notepad = ""

# Configure Gemini AI
API_KEY = "YOUR_GEMINI_API_KEY"  # Store securely in production!
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# TF-IDF Vectorizer Initialization
vectorizer = TfidfVectorizer()

# Function to find the closest task
def find_closest_task(user_query, vectorizer, task_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, task_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    
    if best_match_score > 0.3:  # Similarity threshold
        return df.iloc[best_match_index]["Details"]
    return None

# Google Calendar Integration
def authenticate_google_calendar():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=credentials)

def add_event_to_calendar(summary, description, event_date, event_time, reminder_minutes=30):
    try:
        calendar_service = authenticate_google_calendar()
        
        # Convert to proper datetime format
        start_datetime = f"{event_date}T{event_time}:00"
        end_datetime = (datetime.datetime.strptime(f"{event_date} {event_time}", "%Y-%m-%d %H:%M") + 
                       datetime.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:00")
        
        event = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start_datetime,
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": end_datetime,
                "timeZone": "UTC",
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": reminder_minutes},
                ],
            },
        }
        
        event = calendar_service.events().insert(
            calendarId="primary",
            body=event
        ).execute()
        
        return f"✅ Event added to Google Calendar: {summary} on {event_date} at {event_time}"
    except Exception as e:
        return f"❌ Error adding event: {str(e)}"

# Voice Note Functionality
def record_voice_note(task_title):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording voice note for 5 seconds...")
        audio = r.record(source, duration=5)
        
    try:
        voice_note = r.recognize_google(audio)
        st.session_state.tasks.loc[st.session_state.tasks["Task"] == task_title.lower(), "VoiceNote"] = voice_note
        return f"🎤 Voice note saved for '{task_title}': {voice_note}"
    except Exception as e:
        return f"❌ Error recording voice note: {str(e)}"

# Call Functionality
def make_call(phone_number, task_title):
    try:
        # This uses pywhatkit which requires WhatsApp Web to be open
        kt.sendwhatmsg_instantly(
            phone_no=phone_number,
            message=f"Reminder about task: {task_title}",
            wait_time=15,
            tab_close=True
        )
        time.sleep(5)
        pyautogui.hotkey('ctrl', 'w')  # Close tab after sending
        return f"📞 Call initiated for task: {task_title}"
    except Exception as e:
        return f"❌ Error making call: {str(e)}"

# Reminder Notification
def set_reminder(task, date, time):
    reminder_time = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    current_time = datetime.datetime.now()
    time_diff = (reminder_time - current_time).total_seconds()
    
    if time_diff > 0:
        time.sleep(time_diff)
        notification.notify(
            title="Task Reminder ✅", 
            message=f"Reminder: {task}",
            timeout=10,
            app_icon=None
        )

# Sidebar - Notepad and Quick Actions
st.sidebar.title("📝 Quick Actions")
st.session_state.notepad = st.sidebar.text_area("Notepad:", value=st.session_state.notepad, height=200)

# Main App UI
st.title("Enhanced Task Manager ✅")
st.write("Manage tasks with calendar integration, voice notes, and call reminders.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Task Input Section
with st.expander("➕ Add a New Task", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        task_title = st.text_input("Task Title*")
        task_details = st.text_area("Task Details*")
        task_date = st.date_input("Task Date*", datetime.date.today())
        
    with col2:
        task_time = st.time_input("Task Time*", datetime.time(12, 0))
        add_reminder = st.checkbox("Set Reminder")
        record_voice = st.checkbox("Add Voice Note")
        phone_number = st.text_input("Phone Number (for call reminders)", placeholder="+1234567890")
    
    if st.button("Add Task"):
        if task_title and task_details:
            new_task = pd.DataFrame([[task_title.lower(), task_details.lower(), task_date, 
                                    task_time.strftime("%H:%M"), add_reminder, ""]],
                                  columns=["Task", "Details", "Date", "Time", "Reminder", "VoiceNote"])
            st.session_state.tasks = pd.concat([st.session_state.tasks, new_task], ignore_index=True)
            
            # Calendar Integration
            calendar_msg = add_event_to_calendar(
                task_title, 
                task_details, 
                task_date.strftime("%Y-%m-%d"), 
                task_time.strftime("%H:%M")
            )
            st.success(calendar_msg)
            
            # Voice Note
            if record_voice:
                voice_msg = record_voice_note(task_title.lower())
                st.info(voice_msg)
            
            # Call Reminder
            if phone_number:
                call_msg = make_call(phone_number, task_title)
                st.info(call_msg)
            
            # System Reminder
            if add_reminder:
                set_reminder(task_title, task_date.strftime("%Y-%m-%d"), task_time.strftime("%H:%M"))
                st.success(f"Reminder set for {task_date} at {task_time}")
            
            st.success(f"Task '{task_title}' added successfully!")
        else:
            st.warning("Please fill all required fields (*)")

# Task List Section
st.subheader("📋 Your Tasks")
if not st.session_state.tasks.empty:
    for idx, task in st.session_state.tasks.iterrows():
        with st.expander(f"{task['Task'].title()} - {task['Date']} at {task['Time']}"):
            st.write(f"**Details:** {task['Details']}")
            if task['VoiceNote']:
                st.write(f"**Voice Note:** 🎤 {task['VoiceNote']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Complete ✅", key=f"complete_{idx}"):
                    st.session_state.tasks = st.session_state.tasks.drop(index=idx)
                    st.rerun()
            with col2:
                if st.button(f"Edit ✏️", key=f"edit_{idx}"):
                    # Implement edit functionality
                    pass
            with col3:
                if st.button(f"Call Reminder 📞", key=f"call_{idx}") and "phone_number" in locals():
                    make_call(phone_number, task['Task'])
else:
    st.info("No tasks added yet. Add your first task above!")

# Chatbot Section
st.subheader("💬 Task Assistant")
if prompt := st.chat_input("Ask about tasks or get help..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Search for relevant task
    closest_task = None
    if not st.session_state.tasks.empty:
        task_vectors = vectorizer.fit_transform(st.session_state.tasks["Task"])
        closest_task = find_closest_task(prompt, vectorizer, task_vectors, st.session_state.tasks)

    if closest_task is not None:
        response_text = f"Here's what I found:\n\n**{closest_task}**"
    else:
        # AI-generated response
        try:
            chat = model.start_chat(history=st.session_state.messages)
            response = chat.send_message(prompt)
            response_text = response.text
        except Exception as e:
            response_text = f"Sorry, I couldn't process your request. Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)

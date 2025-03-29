import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import time
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from plyer import notification
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Set up Streamlit Page
st.set_page_config(page_title="Task Manager ✅", page_icon="✅", layout="wide")

# Initialize session state
if "tasks" not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=["Task", "Details", "Date", "Time", "Reminder"])
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
    credentials_path = "YOUR_SERVICE_ACCOUNT_JSON_PATH.json"  # Replace with your Google service account JSON file
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=["https://www.googleapis.com/auth/calendar"])
    return build("calendar", "v3", credentials=creds)

def add_event_to_calendar(summary, description, event_date, event_time):
    try:
        calendar_service = authenticate_google_calendar()
        start_time = f"{event_date}T{event_time}:00"
        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start_time, "timeZone": "UTC"},
            "end": {"dateTime": start_time, "timeZone": "UTC"},
        }
        event = calendar_service.events().insert(calendarId="primary", body=event).execute()
        return f"Event '{summary}' added to Google Calendar on {event_date} at {event_time}."
    except Exception as e:
        return f"Error adding event to Google Calendar: {e}"

# Reminder Notification
def set_reminder(task, date, time):
    reminder_time = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    current_time = datetime.datetime.now()
    time_diff = (reminder_time - current_time).total_seconds()
    
    if time_diff > 0:
        time.sleep(time_diff)
        notification.notify(title="Task Reminder ✅", message=f"Reminder: {task}", timeout=10)

# Sidebar - Notepad
st.sidebar.title("📝 Quick Notepad")
st.session_state.notepad = st.sidebar.text_area("Write your notes here:", value=st.session_state.notepad, height=200)
st.sidebar.button("Save Notes")

# Main App UI
st.title("Personal Task Manager ✅")
st.write("Manage tasks, set reminders, and get AI-powered assistance.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Task Input Section
with st.expander("➕ Add a New Task"):
    task_title = st.text_input("Task Title")
    task_details = st.text_area("Task Details")
    task_date = st.date_input("Task Date")
    task_time = st.time_input("Task Time")
    add_reminder = st.checkbox("Set Reminder")

    if st.button("Add Task"):
        if task_title and task_details:
            new_task = pd.DataFrame([[task_title.lower(), task_details.lower(), task_date, task_time, add_reminder]],
                                    columns=["Task", "Details", "Date", "Time", "Reminder"])
            st.session_state.tasks = pd.concat([st.session_state.tasks, new_task], ignore_index=True)
            st.success(f"Task '{task_title}' added!")

            # Add Event to Google Calendar
            calendar_msg = add_event_to_calendar(task_title, task_details, task_date, task_time)
            st.success(calendar_msg)

            # Set Reminder if checked
            if add_reminder:
                set_reminder(task_title, task_date, task_time)
        else:
            st.warning("Please fill all fields.")

# Train the vectorizer if there are tasks
if not st.session_state.tasks.empty:
    task_vectors = vectorizer.fit_transform(st.session_state.tasks["Task"])

# Chatbot Input
if prompt := st.chat_input("Ask about a task or get AI help..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Search for a relevant task
    closest_task = None
    if not st.session_state.tasks.empty:
        closest_task = find_closest_task(prompt, vectorizer, task_vectors, st.session_state.tasks)

    if closest_task:
        response_text = f"Here’s what I found about your task:\n\n**{closest_task}**"
    else:
        # AI-generated response
        try:
            response = model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            response_text = f"Sorry, I couldn't generate a response. Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)

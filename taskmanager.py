import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up the Streamlit page
st.set_page_config(page_title="Task Manager Chatbot ✅", page_icon="✅", layout="centered")

# Initialize session state for tasks and chat history
if "tasks" not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=["Task", "Details"])
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure Gemini API (Replace with your actual API key)
API_KEY = "YOUR_GEMINI_API_KEY"  # Store in st.secrets or environment variables in production!
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# TF-IDF vectorizer initialization
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

# Streamlit App UI
st.title("Personal Task Manager ✅")
st.write("Manage your tasks and get AI assistance.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Task Input Section
with st.expander("➕ Add a New Task"):
    task_title = st.text_input("Task Title", key="task_title")
    task_details = st.text_area("Task Details", key="task_details")
    if st.button("Add Task"):
        if task_title and task_details:
            new_task = pd.DataFrame([[task_title.lower(), task_details.lower()]], columns=["Task", "Details"])
            st.session_state.tasks = pd.concat([st.session_state.tasks, new_task], ignore_index=True)
            st.success(f"Task '{task_title}' added!")
        else:
            st.warning("Please enter both task title and details.")

# Train the vectorizer if there are tasks
if not st.session_state.tasks.empty:
    task_vectors = vectorizer.fit_transform(st.session_state.tasks["Task"])

# Chatbot Input
if prompt := st.chat_input("Ask about a task or get AI help..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Search for a relevant task
    closest_task = None
    if not st.session_state.tasks.empty:
        closest_task = find_closest_task(prompt, vectorizer, task_vectors, st.session_state.tasks)

    if closest_task:
        response_text = f"Here’s what I found about your task:\n\n**{closest_task}**"
    else:
        # Step 2: Generate a response using Gemini AI
        try:
            response = model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            response_text = f"Sorry, I couldn't generate a response. Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)

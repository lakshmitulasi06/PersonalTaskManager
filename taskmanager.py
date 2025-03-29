import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(page_title="College Chatbot 🎓", page_icon="🎓", layout="centered")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load CSV from GitHub (replace with your actual URL)
csv_url = "college_faq.csv"

try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Failed to load the CSV file. Error: {e}")
    st.stop()

# Preprocess data
df = df.fillna("")
df['Question'] = df['Question'].str.lower()
df['Answer'] = df['Answer'].str.lower()

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

# Configure Gemini API (replace with your actual API key)
API_KEY = "AIzaSyBsq5Kd5nJgx2fejR77NT8v5Lk3PK4gbH8"  # DO NOT EMBED KEYS DIRECTLY! Use st.secrets or env vars!
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to find the closest question using cosine similarity
def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    if best_match_score > 0.3:  # Threshold for similarity
        return df.iloc[best_match_index]['Answer']
    else:
        return None

# Streamlit app
st.title("College Chatbot 🎓")
st.write("Welcome to the College Chatbot! Ask me anything about the college.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Search the CSV file for the closest answer
    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)
    
    if closest_answer:
        # If a relevant answer is found in the CSV, display it directly
        st.session_state.messages.append({"role": "assistant", "content": closest_answer})
        with st.chat_message("assistant"):
            st.markdown(closest_answer)
    else:
        # If no relevant answer is found, use Gemini to generate a concise response
        try:
            response = model.generate_content(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            with st.chat_message("assistant"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Sorry, I couldn't generate a response. Error: {e}")

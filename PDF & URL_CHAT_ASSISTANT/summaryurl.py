import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import pandas as pd

# Set page configuration at the top
st.set_page_config(layout="wide")  # Set wide page layout

# Initialize the model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set your OpenAI API key here
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your actual OpenAI API key

# Function to fetch content from the URL
def fetch_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=200):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to generate embeddings
def generate_embeddings(chunks):
    try:
        embeddings = model.encode(chunks, convert_to_tensor=True)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

# Function to find relevant chunks based on the query
def find_relevant_chunks(query_embedding, chunk_embeddings, chunks):
    cosine_similarities = np.dot(chunk_embeddings, query_embedding.T) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    relevant_indices = np.argsort(cosine_similarities)[-3:][::-1]
    return [(chunks[i], cosine_similarities[i]) for i in relevant_indices]

# Function to get answer from OpenAI model
def get_answer_from_openai(query, relevant_chunks):
    context = " ".join(relevant_chunks)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}],
            max_tokens=150
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error getting response from OpenAI model: {e}"

# Function to prepare data for download
def prepare_downloadable_data(relevant_chunks, user_query, answer):
    data = {
        'User Query': [user_query],
        'Model Answer': [answer],
        'Relevant Chunks': ["\n".join([chunk[0] for chunk in relevant_chunks])]  # Fix: Join only the text part
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# Initialize session state for page management
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []
if 'all_embeddings' not in st.session_state:
    st.session_state.all_embeddings = []
if 'history' not in st.session_state:
    st.session_state.history = []

# Main functionality page
st.markdown("""<style>body { background-color: white; }</style>""", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px; font-weight: bold;'>URLQ&A: Web-Based Query and Answer System</h1>
    <p style='text-align: center; font-size: 20px; font-family: "Italian Font";'>Searching got easy</p>
    """,
    unsafe_allow_html=True
)

# Create input and results area
st.markdown("<p style='color:#50E3C2; font-size: 18px;'>Enter the URL below:</p>", unsafe_allow_html=True)
url = st.text_input("Enter a valid URL", placeholder="Enter a valid URL")

# Fetch content with loading spinner
if st.button("Fetch Content"):
    if url:
        with st.spinner('Fetching content...'):
            content = fetch_url_content(url)
            if "Error" not in content:
                chunks = chunk_text(content, chunk_size=1000)
                chunk_embeddings = generate_embeddings(chunks)

                st.session_state.all_chunks.extend(chunks)
                st.session_state.all_embeddings.extend(chunk_embeddings)
            else:
                st.error(content)
    else:
        st.warning("Please enter a valid URL.")

if st.button("Show History"):
    if st.session_state.history:
        st.write("Query History:")
        for item in st.session_state.history:
            st.write(f"**Query:** {item['query']}\n**Answer:** {item['answer']}")
    else:
        st.write("No history available.")

if st.button("Clear Inputs"):
    st.session_state.all_chunks = []
    st.session_state.all_embeddings = []
    st.session_state.history = []
    st.success("All inputs cleared!")

st.markdown("<p style='color:#50E3C2; font-size: 18px;'>Ask your question:</p>", unsafe_allow_html=True)
user_query = st.text_input("Your Question")

if st.button("Get Answer"):
    if user_query and st.session_state.all_chunks:
        user_query_embedding = generate_embeddings([user_query])[0]
        relevant_chunks = find_relevant_chunks(user_query_embedding, np.array(st.session_state.all_embeddings), st.session_state.all_chunks)
        answer = get_answer_from_openai(user_query, [chunk[0] for chunk in relevant_chunks])
        
        st.session_state.history.append({"query": user_query, "answer": answer})
        st.success(f"Answer: {answer}")

        # Prepare download button
        csv = prepare_downloadable_data(relevant_chunks, user_query, answer)
        st.download_button(label="Download History", data=csv, file_name='query_history.csv', mime='text/csv')
    else:
        st.warning("Please enter a question or fetch content first.")

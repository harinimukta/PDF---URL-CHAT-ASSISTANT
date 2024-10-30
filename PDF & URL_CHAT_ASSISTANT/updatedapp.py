import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI Configuration
st.set_page_config(page_title="PDF & URL Chat Assistant", layout="wide")

# Custom CSS for UI styling
st.markdown("""
    <style>
    body { background-color: #f0f4f8; font-family: 'Arial', sans-serif; }
    .title { text-align: center; font-size: 36px; font-weight: bold; color: #4A90E2; margin-top: 20px; }
    .subtitle { text-align: center; font-size: 20px; color: #333; margin-bottom: 20px; }
    .input-label { font-size: 18px; color: #333; }
    .chat-history { max-height: 400px; overflow-y: auto; background-color: #fff; border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px; }
    .user-message { color: #4A90E2; font-weight: bold; }
    .assistant-message { color: #50E3C2; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📚 PDF & URL Chat Assistant 🌐</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload PDFs or enter a URL to chat with their content</div>", unsafe_allow_html=True)

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    with col2:
        url_input = st.text_input("Or enter a URL:")

    documents = []

    # Process uploaded PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(f"./{uploaded_file.name}", "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(f"./{uploaded_file.name}")
            documents.extend(loader.load())

    # Process URL content
    if url_input:
        try:
            response = requests.get(url_input)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = ' '.join([p.get_text() for p in soup.find_all('p')])
            documents.append(Document(page_content=content))
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching URL: {e}")

    latest_response = None

    if documents:
        # Create document retriever with embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Define retrieval and QA system prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the user's question using the chat history if necessary."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an assistant answering questions based strictly on the provided PDFs or URL content. "
                "If the answer is not available in these documents, respond with: "
                "'This question is not available in the provided PDF or URL.'"
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "{context}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Capture user input
        user_input = st.text_input("Your question:")
        if user_input:
            if len(user_input) > 500:
                st.error("Your question is too long. Please limit it to 500 characters.")
            else:
                session_history = get_session_history(session_id)
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input, "context": ""},  
                        config={"configurable": {"session_id": session_id}},
                    )
                    latest_response = response['answer']

                    # Enforce strict response from provided documents only
                    if not latest_response.strip() or latest_response == "This question is not available in the provided PDF or URL.":
                        latest_response = "This question is not available in the provided PDF or URL."
                    
                    st.session_state.chat_history.append({"user": user_input, "assistant": latest_response})
                except Exception as e:
                    st.error(f"Error in processing the request: {e}")

    with st.sidebar:
        st.markdown("### Chat History", unsafe_allow_html=True)
        for message in st.session_state.chat_history[:-1]:
            st.markdown(f"**You:** {message['user']}", unsafe_allow_html=True)
            with st.expander("Assistant's response", expanded=False):
                st.markdown(f"*{message['assistant']}*", unsafe_allow_html=True)

    if latest_response:
        st.markdown(f"**Latest Assistant Response:** *{latest_response}*")

else:
    st.warning("Please ensure the GROQ_API_KEY is set in the .env file.")

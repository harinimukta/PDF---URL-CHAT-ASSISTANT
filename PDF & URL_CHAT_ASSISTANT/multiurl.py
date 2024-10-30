import streamlit as st
import requests
from pytube import YouTube
from moviepy.editor import VideoFileClip
import speech_recognition as sr
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
import os
from dotenv import load_dotenv

load_dotenv()

# Set HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to transcribe video
def transcribe_video(video_path):
    recognizer = sr.Recognizer()
    audio_clip = VideoFileClip(video_path).audio
    audio_clip.write_audiofile("temp_audio.wav")
    
    with sr.AudioFile("temp_audio.wav") as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

# Set up Streamlit
st.title("Conversational RAG With YouTube Video Content")
youtube_link = st.text_input("Enter YouTube Video URL:")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key and youtube_link:
    try:
        # Download video from YouTube
        st.write("Downloading video...")
        yt = YouTube(youtube_link)
        video_stream = yt.streams.filter(file_extension='mp4').first()
        video_path = video_stream.download(filename="temp_video.mp4")
        
        # Transcribe the video
        st.write("Transcribing video...")
        text_content = transcribe_video(video_path)
        st.write("Transcribed Content:")
        st.write(text_content)

        # Prepare for RAG
        all_documents = [Document(page_content=text_content)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(all_documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Chat interface
        session_id = st.text_input("Session ID", value="default_session")

        # Statefully manage chat history
        if 'store' not in st.session_state:
            st.session_state.store = {}

        # Define chat prompts
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "reformulate it if needed."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

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

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke({"input": user_input})
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
    
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please enter the Groq API Key and a YouTube link.")

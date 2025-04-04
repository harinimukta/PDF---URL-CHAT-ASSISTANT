📚 PDF & URL Chat Assistant 🌐
A Streamlit-based AI assistant that allows users to upload PDF documents or enter a URL, and interactively chat with the content using a Groq LLM backend. Includes support for voice input and text-to-speech output, making it highly accessible and user-friendly.

✨ Features
✅ Upload and chat with multiple PDF files

✅ Fetch and chat with content from a public URL

✅ Voice input (speech-to-text using Google Speech Recognition)

✅ Text-to-speech responses using pyttsx3

✅ Conversational memory to understand question context

✅ Powered by Groq (Gemma2-9b-It) for fast and intelligent responses

✅ Visual chat history for past questions and answers

✅ Handles long document content via chunking and embedding

🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/pdf-url-chat-assistant.git](https://github.com/harinimukta/PDF---URL-CHAT-ASSISTANT.git
cd pdf-url-chat-assistant
2. Set Up a Virtual Environment
bash
Copy
Edit
python -m venv bgenv
source bgenv/bin/activate       # On Windows: bgenv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Set Up Environment Variables
Create a .env file in the project root:

env
Copy
Edit
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
🔐 Both GROQ_API_KEY and HF_TOKEN are required for LLM and embeddings.

🧠 How It Works
PDF works using RAG concept
PDF/URL Loader: Extracts text from uploaded PDFs or HTML content from given URLs.

Text Splitter: Splits content into smaller chunks for better semantic search.

Embeddings: Converts document chunks into vector embeddings using HuggingFace.

Chroma Vector Store: Stores and retrieves relevant document chunks using semantic similarity.

Groq LLM: Uses Gemma2-9b-It to generate context-aware responses based on retrieved content.

LangChain: Chains multiple components together with history-aware context tracking.

🎙️ Voice Assistant Integration
Uses speech_recognition to capture your voice through the microphone.

Converts speech to text and sends it to the assistant.

Reads the assistant's response aloud using pyttsx3.

Streamlit – For UI and interaction

LangChain – For chaining prompts and handling chat memory

Groq (Gemma2-9b-It) – As the LLM backend

Hugging Face Transformers – For embeddings

Chroma – For vector store and document retrieval

SpeechRecognition & pyttsx3 – For voice input and output

BeautifulSoup & Requests – For URL content parsing

✅ To-Do / Future Enhancements
 Support for other document types (DOCX, TXT)

 Multi-language voice input/output

 Mobile-friendly UI design

 Upload content via Google Drive or Dropbox


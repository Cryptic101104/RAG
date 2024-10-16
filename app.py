import streamlit as st
import fitz  # PyMuPDF for PDF handling
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber

# Custom CSS for layout and design
st.markdown(
    """
    <style>
    /* Hide Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Left Sidebar (History) */
    .sidebar .sidebar-content {
        background-color: #202030;  /* Dark background to match the app theme */
        padding: 20px;
        color: white;
    }

    /* Styling for the question history */
    .history-box {
        background-color: #303040;  /* Darker background to blend in */
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #444455;  /* Lighter border for subtle contrast */
        color: white;  /* Text color */
    }

    /* Bottom bar (Question input) */
    .bottom-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f8f9fa;
        padding: 10px;
        border-top: 1px solid #e0e0e0;
    }

    /* Input box inside the bottom bar */
    .bottom-bar input[type="text"] {
        width: 80%;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #d3d3d3;
    }

    /* Submit button inside the bottom bar */
    .bottom-bar button {
        width: 15%;
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px;
        cursor: pointer;
    }

    /* Main content styling */
    .main-content {
        padding-bottom: 70px; /* Space for bottom bar */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load the embedding model (optimized for CPU)
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Function to extract text from PDF (using PyMuPDF)
import streamlit as st
import fitz  # PyMuPDF for PDF handling
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber

# Custom CSS for layout and design
st.markdown(
    """
    <style>
    /* Hide Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Left Sidebar (History) */
    .sidebar .sidebar-content {
        background-color: #202030;  /* Dark background to match the app theme */
        padding: 20px;
        color: white;
    }

    /* Styling for the question history */
    .history-box {
        background-color: #303040;  /* Darker background to blend in */
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #444455;  /* Lighter border for subtle contrast */
        color: white;  /* Text color */
    }

    /* Bottom bar (Question input) */
    .bottom-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f8f9fa;
        padding: 10px;
        border-top: 1px solid #e0e0e0;
    }

    /* Input box inside the bottom bar */
    .bottom-bar input[type="text"] {
        width: 80%;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #d3d3d3;
    }

    /* Submit button inside the bottom bar */
    .bottom-bar button {
        width: 15%;
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px;
        cursor: pointer;
    }

    /* Main content styling */
    .main-content {
        padding-bottom: 70px; /* Space for bottom bar */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load the embedding model (optimized for CPU)
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to preprocess and split text into smaller chunks
def preprocess_text(text, chunk_size=5):
    sentences = text.split('. ')
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

# Function to create embeddings
def create_embeddings(text_chunks):
    embeddings = model.encode(text_chunks)
    return np.array(embeddings)

# Function to initialize FAISS index
def initialize_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

# Store question history in session state
if 'question_history' not in st.session_state:
    st.session_state.question_history = []

# Main Streamlit App
st.title("RAG (Retrieval-Augmented Generation) Web App")
st.write("Upload a PDF, and ask questions based on its content.")

# Left Sidebar (History of Questions and Answers)
with st.sidebar:
    st.subheader("📜 Question History")
    for idx, q in enumerate(st.session_state.question_history):
        with st.expander(f"Q: {q['question']}", expanded=False):
            st.write(f"**A:** {q['answer']}")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner('Processing your document...'):
        extracted_text = extract_text_from_pdf(uploaded_file)
        text_chunks = preprocess_text(extracted_text)
        embeddings = create_embeddings(text_chunks)
        faiss_index = initialize_faiss_index(embeddings)
    st.success("PDF processed and indexed. You can now ask questions.")

# Get user input
user_query = st.text_input("Enter your question below:", key="question")

if user_query:
    with st.spinner('Searching for relevant passages...'):
        query_embedding = model.encode([user_query])
        D, I = faiss_index.search(np.array(query_embedding), k=5)  # Get top 5 results
        
        st.write("Top relevant passages:")
        results = []
        for idx in I[0]:
            passage = text_chunks[idx]
            st.write(f"Passage {idx + 1}: {passage}")
            results.append(passage)

        # Add to question history (show top 3 results)
        st.session_state.question_history.append({
            "question": user_query,
            "answer": ", ".join(results[:3])  # Display top 3 results in the history
        })



# Function to preprocess and split text into smaller chunks
def preprocess_text(text, chunk_size=5):
    sentences = text.split('. ')
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

# Function to create embeddings
def create_embeddings(text_chunks):
    embeddings = model.encode(text_chunks)
    return np.array(embeddings)

# Function to initialize FAISS index
def initialize_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

# Store question history in session state
if 'question_history' not in st.session_state:
    st.session_state.question_history = []

# Main Streamlit App
st.title("RAG (Retrieval-Augmented Generation) Web App")
st.write("Upload a PDF, and ask questions based on its content.")

# Left Sidebar (History of Questions and Answers)
with st.sidebar:
    st.subheader("📜 Question History")
    for idx, q in enumerate(st.session_state.question_history):
        with st.expander(f"Q: {q['question']}", expanded=False):
            st.write(f"**A:** {q['answer']}")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner('Processing your document...'):
        extracted_text = extract_text_from_pdf(uploaded_file)
        text_chunks = preprocess_text(extracted_text)
        embeddings = create_embeddings(text_chunks)
        faiss_index = initialize_faiss_index(embeddings)
    st.success("PDF processed and indexed. You can now ask questions.")

# Get user input
user_query = st.text_input("Enter your question below:", key="question")

if user_query:
    with st.spinner('Searching for relevant passages...'):
        query_embedding = model.encode([user_query])
        D, I = faiss_index.search(np.array(query_embedding), k=5)  # Get top 5 results
        
        st.write("Top relevant passages:")
        results = []
        for idx in I[0]:
            passage = text_chunks[idx]
            st.write(f"Passage {idx + 1}: {passage}")
            results.append(passage)

        # Add to question history (show top 3 results)
        st.session_state.question_history.append({
            "question": user_query,
            "answer": ", ".join(results[:3])  # Display top 3 results in the history
        })


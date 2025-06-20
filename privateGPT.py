import os
import time
import base64
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

import streamlit as st

# Set User-Agent to mimic Brave
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"

# Set up environment variables
model = os.environ.get("MODEL", "llama3.1")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 8))

# Load documents from the MUJ website with specific pages
urls = [
    "https://jaipur.manipal.edu/",
    "https://jaipur.manipal.edu/muj-admissions.html",
    "https://jaipur.manipal.edu/muj-academics.html",
    "https://jaipur.manipal.edu/muj-campus-life.html"
]
loader = WebBaseLoader(urls)
documents = loader.load()

# Initialize embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={"device": device})

# Initialize the LLM
llm = OllamaLLM(base_url="http://127.0.0.1:11435", model=model)

# Define prompt template
prompt_template = """You are a knowledgeable assistant for Manipal University Jaipur (MUJ). Your role is to provide accurate and helpful information about the university, its programs, campus life, facilities, and academic opportunities.

Use the following context to answer the question. If the context doesn't provide enough information, you may use your general knowledge about universities and education, but clearly indicate when you're doing so. If you don't know the answer, be honest and say so. Provide the answer in at least 30 words if you don't know the answer use your knowledge to do so.

Please ensure your responses are:
1. Detailed (at least 30-40 words)
2. Well-structured
3. Focused on providing valuable information to students
4. Professional yet friendly in tone

Context: {context}

Question: {question}

Detailed Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def ask_question(query: str, qa_chain) -> str:
    """
    Process a question and return the answer using the QA chain
    """
    result = qa_chain.invoke({"query": query})
    return result['result']

# Function to load SVG as base64
def get_svg_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            svg_bytes = f.read()
        return base64.b64encode(svg_bytes).decode()
    return None

# Function to render chat messages
def render_message(role, content):
    message_class = "user-message" if role == "user" else "assistant-message"
    st.markdown(f"""
        <div class="chat-message {message_class}">
            <b>{'You:' if role == 'user' else 'Assistant:'}</b> {content}
        </div>
    """, unsafe_allow_html=True)

# Streamlit UI
def main():
    # Set page configuration
    st.set_page_config(
        page_title="MUJ Student Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS - Dark theme inspired by WhatsApp chat wallpaper
    st.markdown("""
        <style>
        /* Main background and app container - WhatsApp-like dark theme */
        .stApp {
            background-color: #1f2c34; /* Dark gray similar to WhatsApp default */
            color: #ffffff; /* White text for contrast */
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #ffffff;
        }
        
        /* Sidebar styling - making it darker to match */
        section[data-testid="stSidebar"] {
            background-color: #1a252d; /* Slightly lighter than main for contrast */
            color: #ffffff;
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #ffffff;
        }
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #4da6ff;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #2c5282;
            color: #ffffff;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: #3873b3;
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.24);
        }
        
        .user-message {
            background-color: #ADD8E6; /* Light blue for user input */
            border-left: 4px solid #1f4d7a;
            color: #333333; /* Darker text for user input */
        }
        
        .assistant-message {
            background-color: #F8E0E6; /* Pinkish for assistant response */
            border-left: 4px solid #7b1fa2;
            color: #333333; /* Darker text for assistant response */
        }
        
        /* Input box styling */
        .stChatInputContainer {
            border-radius: 10px;
            background-color: #2a3b45; /* Darker input background to match theme */
            color: #ffffff;
        }
        
        .stChatInputContainer input {
            color: #ffffff;
        }
        
        /* Custom MUJ header banner */
        .muj-header {
            background-color: #2c5282;
            color: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        /* MUJ header logo */
        .muj-header img {
            margin-bottom: 1rem;
        }
        
        /* MUJ header text */
        .muj-header h2 {
            margin-bottom: 0.5rem;
        }
        
        /* Styling for sidebar section titles */
        .sidebar-title {
            color: #4da6ff !important;
            font-size: 1.2rem;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* System info styling */
        .system-info {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #3a4b58;
            font-size: 0.85rem;
            opacity: 0.8;
        }
        
        /* Chat input placeholder */
        .stTextInput input::placeholder {
            color: #a0b0c0; /* Lighter gray for placeholder */
        }
        </style>
    """, unsafe_allow_html=True)

    # Try to load SVG as base64
    svg_base64 = get_svg_base64("manipal-university-jaipur-logo.svg")

    # Header with MUJ logo and welcome message
    if svg_base64:
        st.markdown(f"""
            <div class="muj-header">
                <img src="data:image/svg+xml;base64,{svg_base64}" width="400" alt="MUJ Logo" />
                <h2>Welcome to the Manipal University Jaipur Student Assistant!</h2>
                <p>I'm here to help you with any questions about MUJ.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="muj-header">
                <h2>Welcome to the Manipal University Jaipur Student Assistant!</h2>
                <p>I'm here to help you with any questions about MUJ.</p>
            </div>
        """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            render_message(message["role"], message["content"])

    # Load documents and set up database
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if documents:
        db.add_documents(documents)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Initialize the QA chain with the retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Chat input
    if prompt := st.chat_input("Ask me anything about MUJ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.container():
            render_message("user", prompt)

        with st.spinner("Thinking..."):
            response = ask_question(prompt, qa_chain)

            # Simulate typing effect
            with st.container():
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(f"""
                        <div class="chat-message assistant-message">
                            <b>Assistant:</b> {full_response}▌
                        </div>
                    """, unsafe_allow_html=True)

                message_placeholder.markdown(f"""
                    <div class="chat-message assistant-message">
                        <b>Assistant:</b> {full_response}
                    </div>
                """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with improved styling
    with st.sidebar:
        # About section
        st.markdown('<p class="sidebar-title">About MUJ Assistant</p>', unsafe_allow_html=True)

        st.markdown("""
            This AI assistant is designed to help students learn about Manipal University Jaipur.
            Feel free to ask questions about:
            - Academic programs
            - Campus facilities  
            - Admission process
            - Student life
            - Research opportunities
            - And more!
        """)

        # Settings section
        st.markdown('<p class="sidebar-title">Settings</p>', unsafe_allow_html=True)

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()  # Updated to use st.rerun()

        # System information section
        st.markdown('<p class="sidebar-title">System Information</p>', unsafe_allow_html=True)

        st.markdown(f"""
            <div class="system-info">
                Model: {model}<br>
                Embeddings: {embeddings_model_name}
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
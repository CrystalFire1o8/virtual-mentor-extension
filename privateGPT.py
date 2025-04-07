import os
import time
import base64
from typing import Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

import streamlit as st

# Set up environment variables
model = os.environ.get("MODEL", "llama3.1")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 8))

# Initialize the embeddings and database
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

# Initialize the LLM
llm = Ollama(model=model)

# Define prompt template
prompt_template = """You are a knowledgeable assistant for Manipal University Jaipur (MUJ). Your role is to provide accurate and helpful information about the university, its programs, campus life, facilities, and academic opportunities.

Use the following context to answer the question. If the context doesn't provide enough information, you may use your general knowledge about universities and education, but clearly indicate when you're doing so. If you don't know the answer, be honest and say so.Provide the answer in atleast 30 words if you dont know the answer use your knowldege to do so.

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

# Initialize the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

def ask_question(query: str) -> str:
    """
    Process a question and return the answer using the QA chain
    """
    result = qa(query)
    return result['result']

# Function to load SVG as base64
def get_svg_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            svg_bytes = f.read()
        return base64.b64encode(svg_bytes).decode()
    return None

# Streamlit UI
def main():
    # Set page configuration
    st.set_page_config(
        page_title="MUJ Student Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS - Fixed to match dark theme
    st.markdown("""
        <style>
        /* Main background and app container */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #fff;
        }
        
        /* Sidebar styling - making it dark */
        section[data-testid="stSidebar"] {
            background-color: #1e2130;
            color: white;
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: white;
        }
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #4da6ff;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #2c5282;
            color: white;
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
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }
        
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #1f4d7a;
        }
        
        .assistant-message {
            background-color: #f3e5f5;
            border-left: 4px solid #7b1fa2;
        }
        
        /* Input box styling */
        .stChatInputContainer {
            border-radius: 10px;
            background-color: #f1f3f4;
        }
        
        /* Custom MUJ header banner */
        .muj-header {
            background-color: #2c5282;
            color: white;
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
            border-top: 1px solid #3a3f4b;
            font-size: 0.85rem;
            opacity: 0.8;
        }
        
        /* Chat input placeholder */
        .stTextInput input::placeholder {
            color: #5a6070;
        }
        </style>
    """, unsafe_allow_html=True)

    # Try to load SVG as base64
    svg_base64 = get_svg_base64("manipal-university-jaipur-logo.svg")

    # Header with MUJ logo and welcome message
    if svg_base64:
        # Display with base64 encoded SVG
        st.markdown(f"""
            <div class="muj-header">
                <img src="data:image/svg+xml;base64,{svg_base64}" width="400" alt="MUJ Logo" />
                <h2>Welcome to the Manipal University Jaipur Student Assistant!</h2>
                <p>I'm here to help you with any questions about MUJ.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback if SVG file is not found
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
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>You:</b> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <b>Assistant:</b> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask me anything about MUJ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.container():
            st.markdown(f"""
                <div class="chat-message user-message">
                    <b>You:</b> {prompt}
                </div>
            """, unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            response = ask_question(prompt)

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

    # Sidebar with improved styling matching the screenshot
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
            st.experimental_rerun()

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
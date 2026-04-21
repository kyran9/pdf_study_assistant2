import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
import os

# App Config
st.set_page_config(page_title="Study Assistant", page_icon="📚")

st.title(" Llama 3.2 Study Assistant")
st.caption("Running 100% locally via Ollama")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload Notes")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        st.success("PDF Received!")

# Main Logic
if uploaded_file:
    # Save the uploaded file locally so LangChain can read it
    with open("temp_study_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and extract text
    loader = PyPDFLoader("temp_study_file.pdf")
    pages = loader.load()
    
    # Combine all pages into one text block (Llama 3.2 can handle a lot!)
    full_text = "\n".join([p.page_content for p in pages])
    
    # Chat Input
    user_question = st.text_input("What would you like to know about these notes?")

    if user_question:
        # Initialize the local LLM
        llm = ChatOllama(model="llama3.2")
        
        # Build the prompt
        prompt = f"""
        You are a helpful study assistant. Use the following notes to answer the student's question.
        If the answer isn't in the notes, say you aren't sure based on the provided material.
        
        NOTES:
        {full_text[:10000]} # Sending first 10k characters to keep it snappy
        
        QUESTION:
        {user_question}
        """

        with st.spinner("Analyzing your notes..."):
            response = llm.invoke(prompt)
            st.markdown("### 📝 Answer:")
            st.write(response.content)

else:
    st.info("Please upload a PDF in the sidebar to begin.")

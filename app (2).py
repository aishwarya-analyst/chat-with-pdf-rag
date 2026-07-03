
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# -----------------------------
# Enter your Gemini API Key here
# -----------------------------
os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_KEY_HERE"

st.set_page_config(page_title="Chat with your PDF")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    st.success(f"PDF processed into {len(chunks)} chunks")

    # Create embeddings (UPDATED MODEL)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    # Retrieval QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # User Query
    query = st.text_input("Ask a question about your PDF")

    if query:
        with st.spinner("Thinking..."):
            answer = qa.run(query)

        st.subheader("Answer")
        st.write(answer)


import streamlit as st
from rag_utils import load_documents, split_documents, create_faiss_index, load_faiss_index, build_rag_chain
import os
import tempfile

st.set_page_config(page_title="Excel RAG Agent", layout="wide")
st.title("🧠 RAG Agent for Measurement Data")

uploaded_file = st.file_uploader("📤 Upload your Excel, TXT, or PDF file", type=["xlsx", "xls", "txt", "pdf", "docx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.success(f"File uploaded: {uploaded_file.name}")
    
    with st.spinner("🔍 Loading and processing document..."):
        docs = load_documents(tmp_file_path, uploaded_file.name)
        chunks = split_documents(docs)
        db = create_faiss_index(chunks, persist_dir="vector_db")
        rag_chain = build_rag_chain(db)
    
    st.success("✅ Document processed and indexed!")

    question = st.text_input("❓ Ask a question about the data:")
    
    if question:
        with st.spinner("💬 Generating answer..."):
            result = rag_chain(question)
            st.markdown("### 🧠 Answer")
            st.write(result["result"])
            
            st.markdown("### 📚 Source Chunks")
            for doc in result["source_documents"]:
                st.markdown(f"• `{doc.metadata.get('source', 'unknown')}`")
                st.code(doc.page_content[:500], language="text")

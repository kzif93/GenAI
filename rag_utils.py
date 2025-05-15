
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

def load_documents(file_path, original_filename=None):
    if original_filename is None:
        original_filename = file_path

    if original_filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif original_filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif original_filename.endswith(".xlsx") or original_filename.endswith(".xls"):
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_faiss_index(docs, persist_dir="vector_db"):
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(persist_dir)
    return vectordb

def load_faiss_index(persist_dir="vector_db"):
    embeddings = get_embeddings()
    return FAISS.load_local(persist_dir, embeddings)

def load_llama_model(model_path="./models/mistral.Q4_K_M.gguf"):
    return LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        temperature=0.1,
        top_p=0.95,
        n_gpu_layers=-1,
        verbose=False
    )

def build_rag_chain(vectordb, model_path="./models/mistral.Q4_K_M.gguf"):
    llm = load_llama_model(model_path)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

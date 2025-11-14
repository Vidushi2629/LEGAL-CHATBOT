from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Directory where PDFs and FAISS DB are stored
pdfs_directory = "pdfs/"
FAISS_DB_PATH = "vectorstore/db_faiss"

# Step 1: Upload PDF
def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path  # Return saved file path

# Step 2: Load the uploaded PDF
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Step 3: Create text chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

#  Step 4: Get embedding model
embedding_model_name = "nomic-embed-text"

def get_embedding_model(model_name):
    return OllamaEmbeddings(model=model_name)

# Step 5: Build FAISS DB dynamically from uploaded PDF
def build_faiss_from_pdf(file):
    file_path = upload_pdf(file)
    documents = load_pdf(file_path)
    text_chunks = create_chunks(documents)
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(embedding_model_name))
    faiss_db.save_local(FAISS_DB_PATH)
    print(f" FAISS database updated for: {file.name}")
    
    return faiss_db

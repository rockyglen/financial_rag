# ingest.py

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Using OpenAI for Cloud deployment
from langchain_community.vectorstores import Chroma
import os

# --- 1. CONFIGURATION ---
DATA_PATH = "./data"
# IMPORTANT: This folder must be copied to the EC2 instance
CHROMA_PATH = "./chroma_db_live"
EMBEDDING_MODEL = "text-embedding-ada-002"


# --- 2. LOAD DOCUMENTS (Robust HTML/HTM Loader) ---
def load_all_documents(data_path=DATA_PATH):
    documents = []

    # Check if the data folder exists before listing files
    if not os.path.exists(data_path):
        print(f"Error: Data directory {data_path} not found.")
        return []

    all_files = os.listdir(data_path)
    print(f"Files found in ./data: {all_files}")

    for filename in all_files:
        if filename.startswith("."):
            continue

        file_path = os.path.join(data_path, filename)
        if filename.lower().endswith((".html", ".htm")):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            continue
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return documents


print("1. Loading documents...")
documents = load_all_documents()

# --- 3. CHUNKING ---
if not documents:
    print("No documents were loaded. Stopping ingestion.")
    exit()

print("2. Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks.")

# --- 4. EMBEDDING AND STORAGE ---
# Ensure OPENAI_API_KEY is set as an environment variable before running this locally
print(f"3. Initializing OpenAI embedding model...")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

print(f"4. Creating ChromaDB at {CHROMA_PATH} and storing embeddings...")
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
)
vectorstore.persist()
print("Ingestion complete. ChromaDB saved to disk.")

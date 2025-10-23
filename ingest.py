# ingest.py

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

# --- 1. CONFIGURATION ---
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"


# --- 2. LOAD DOCUMENTS (Robust HTML/HTM Loader) ---
def load_all_documents(data_path):
    """Loads documents by explicitly iterating over files and assigning loaders."""

    all_files = os.listdir(data_path)
    documents = []

    print(f"Files found in ./data: {all_files}")

    for filename in all_files:
        if filename.startswith("."):
            continue

        file_path = os.path.join(data_path, filename)

        # Use UnstructuredHTMLLoader for files ending in .html or .htm
        if filename.lower().endswith((".html", ".htm")):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            print(f"Skipping file: {filename} (Unsupported extension)")
            continue

        try:
            documents.extend(loader.load())
            print(f"Successfully loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents


print("1. Loading documents...")
documents = load_all_documents(DATA_PATH)
print(f"Loaded {len(documents)} document pages/files (total chunks before splitting).")


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
print(f"3. Initializing Ollama embedding model: {EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

print(f"4. Creating ChromaDB at {CHROMA_PATH} and storing embeddings...")
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
)
vectorstore.persist()
print("Ingestion complete. ChromaDB saved to disk.")

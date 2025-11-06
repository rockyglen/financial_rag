# ingest.py

from langchain_community.document_loaders import (
    UnstructuredHTMLLoader,
)  # Imports loader for HTML/HTM files.
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)  # Imports the semantic text splitter.
from langchain_openai import (
    OpenAIEmbeddings,
)  # Imports OpenAI's model to convert text to vectors.
from langchain_community.vectorstores import (
    Chroma,
)  # Imports the local vector database.
import os  # Standard library for interacting with the operating system (paths).

# --- 1. CONFIGURATION ---
DATA_PATH = "./data"  # Defines the input directory where filings are stored.
CHROMA_PATH = (
    "./chroma_db_live"  # Defines the output directory for the vector database.
)
EMBEDDING_MODEL = "text-embedding-ada-002"  # Specifies the OpenAI model for embeddings.


# --- 2. LOAD DOCUMENTS (Robust HTML/HTM Loader) ---
def load_all_documents(data_path=DATA_PATH):
    """Loads documents by explicitly iterating over files and assigning loaders."""

    documents = []  # Initializes an empty list to hold loaded documents.
    if not os.path.exists(data_path):
        print(f"Error: Data directory {data_path} not found.")
        return []  # Returns an empty list if the folder is missing.

    all_files = os.listdir(data_path)  # Lists all files inside the data folder.

    for filename in all_files:
        if filename.startswith("."):
            continue  # Skips hidden files (like .DS_Store).

        file_path = os.path.join(
            data_path, filename
        )  # Creates the full path to the file.

        # Checks if the file ends with .html or .htm (case-insensitive).
        if filename.lower().endswith((".html", ".htm")):
            loader = UnstructuredHTMLLoader(
                file_path
            )  # Creates a specific loader for HTML.
        else:
            continue  # Skips any file that isn't HTML/HTM.

        try:
            documents.extend(
                loader.load()
            )  # Loads the document and adds it to the list.
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return documents


print("1. Loading documents...")
documents = load_all_documents()  # Calls the function to start loading.

# --- 3. CHUNKING ---
if not documents:
    print("No documents were loaded. Stopping ingestion.")
    exit()  # Stops the script if no files were found.

print("2. Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Max size of each text block (chunk).
    chunk_overlap=200,  # Number of characters to overlap between adjacent chunks (maintains context).
    separators=[
        "\n\n",
        "\n",
        " ",
        "",
    ],  # Characters used to try and split text semantically.
)
chunks = text_splitter.split_documents(documents)  # Performs the splitting.

# --- 4. EMBEDDING AND STORAGE ---
# Assumes OPENAI_API_KEY environment variable is set.
print(f"3. Initializing OpenAI embedding model...")
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)  # Initializes the embedding model client.

print(f"4. Creating ChromaDB at {CHROMA_PATH} and storing embeddings...")
vectorstore = Chroma.from_documents(
    documents=chunks,  # The text chunks to store.
    embedding=embeddings,  # The embedding model used to create the vectors.
    persist_directory=CHROMA_PATH,  # The directory where the database files are saved.
)
# vectorstore.persist() # Chroma 0.4.x+ automatically persists, but this is kept for compatibility.
print("Ingestion complete. ChromaDB saved to disk.")

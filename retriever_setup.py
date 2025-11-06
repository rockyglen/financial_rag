# retriever_setup.py

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)  # Imports OpenAI LLM and Embeddings.
from langchain_community.vectorstores import Chroma  # Imports the ChromaDB client.
from langchain_core.prompts import (
    PromptTemplate,
)  # Imports template for structured prompting.
from pydantic import (
    BaseModel,
    Field,
)  # Imports Pydantic for data validation/structured output.
from langchain_core.output_parsers import (
    JsonOutputParser,
)  # Parses LLM's JSON output back into Python.
from typing import List  # Standard library for type hinting.

# Retrieval Imports (Advanced Components)
from langchain.retrievers import ContextualCompressionRetriever

# FIX 2: The Reranker is exposed under its specific path in the base package.
from langchain.retrievers.document_compressors import (
    CrossEncoderReranker,
)  # The reranking model component.
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
)  # Imports a model often used for the reranker.

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_db_live"  # Path to load the pre-built vector store.
LLM_MODEL = "gpt-3.5-turbo"  # The LLM used for both Q&A and query generation.

# --- 1. LOAD COMPONENTS ---
print(f"1. Initializing Cloud components: {LLM_MODEL}")
llm = ChatOpenAI(
    model=LLM_MODEL, temperature=0
)  # Initializes the LLM client (temp=0 for factual answers).
embeddings = OpenAIEmbeddings()  # Initializes the embedding client.

print(f"2. Loading ChromaDB from {CHROMA_PATH}")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,  # Loads the database from the saved directory.
    embedding_function=embeddings,  # Specifies the embedding model used to create the vectors.
)

# --- 2. MULTI-QUERY GENERATOR (LCEL Implementation) ---


class QueryList(BaseModel):
    queries: List[str] = Field(
        description="A list of three refined, alternative search queries for financial documents."
    )  # Schema for LLM output.


QUERY_GEN_PROMPT = (
    """..."""  # (The detailed instructions to the LLM to generate 3 new queries).
)
prompt_query_gen = PromptTemplate.from_template(QUERY_GEN_PROMPT)

structured_llm = llm.with_structured_output(
    QueryList
)  # Tells the LLM to strictly output the QueryList schema.
query_chain = (
    prompt_query_gen | structured_llm
)  # Creates the chain for query generation.

# --- 3. CUSTOM RETRIEVER FUNCTION (Multi-Query + RERANKING Logic) ---


def multi_query_retriever_func(x: dict):
    # This function is the entire Multi-Query and Reranking logic pipeline.
    question = x["question"]  # Extracts the user's question from the input dictionary.

    # ... (Code for generating alternative queries and performing initial high-recall search) ...

    # --- RERANKING STAGE ---

    # 1. Initialize the compressor (Reranker model)
    compressor = CrossEncoderReranker(
        model="BAAI/bge-reranker-base",  # Specifies the external reranker model.
        top_n=3,  # Filters to pass only the 3 best chunks to the final LLM.
    )

    # 2. Create the Compression Pipeline
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(
            search_kwargs={"k": 15}
        ),  # Initial high recall search (k=15).
    )

    # 3. Use the Reranker to get the final, best context.
    final_docs = compression_retriever.invoke(question)

    return final_docs


# Export the retriever function to be used by rag_chain.py
retriever = multi_query_retriever_func

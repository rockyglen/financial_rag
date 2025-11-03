# retriever_setup.py

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List

# Retrieval Imports (Stable Paths)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# --- CONFIGURATION (Uses deployed structure) ---
CHROMA_PATH = "./chroma_db_live"
LLM_MODEL = "gpt-3.5-turbo"

# --- 1. LOAD COMPONENTS ---
print(f"1. Initializing Cloud components: {LLM_MODEL}")
# LLM and Embeddings pull API Key from the OS environment (AWS/EC2)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
embeddings = OpenAIEmbeddings()

print(f"2. Loading ChromaDB from {CHROMA_PATH}")
# The EC2 instance will load this directory
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# --- 2. MULTI-QUERY GENERATOR (LCEL IMPLEMENTATION) ---


class QueryList(BaseModel):
    queries: List[str] = Field(
        description="A list of three refined, alternative search queries for financial documents."
    )


QUERY_GEN_PROMPT = """
You are a specialized query generator for a financial search engine. 
Your task is to analyze the user's complex question and generate 3 alternative search queries 
to retrieve the maximum amount of relevant context from financial documents. 
Original Question: {question}
"""
prompt_query_gen = PromptTemplate.from_template(QUERY_GEN_PROMPT)
structured_llm = llm.with_structured_output(QueryList)
query_chain = prompt_query_gen | structured_llm

# --- 3. CUSTOM RETRIEVER FUNCTION (Multi-Query + RERANKING Logic) ---


def multi_query_retriever_func(x: dict):
    question = x["question"]
    print(f"   [Retrieval] Optimizing query: {question}")

    # a. Generate alternative queries (using OpenAI)
    try:
        generated_queries_pydantic = query_chain.invoke({"question": question})
        all_queries = [question] + generated_queries_pydantic.queries
    except Exception as e:
        print(
            f"   [WARNING] Query generation failed ({e}). Falling back to single query."
        )
        all_queries = [question]

    # b. Perform parallel search (initial high recall search)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # --- RERANKING STAGE (The precision improvement) ---

    # 1. Initialize the compressor (Reranker)
    compressor = CrossEncoderReranker(model="BAAI/bge-reranker-base", top_n=3)

    # 2. Create the Compression Pipeline
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # 3. Use the Reranker on the original question to get the final, best context
    final_docs = compression_retriever.invoke(question)

    print(f"   [Retrieval] Final Reranked Context Size: {len(final_docs)} chunks.")
    return final_docs


# Export the retriever function
retriever = multi_query_retriever_func

print("Retriever setup complete. The retriever is a function ready for execution.")

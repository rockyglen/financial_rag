# rag_chain.py

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from retriever_setup import (
    retriever,
    llm,
)  # Import the functional retriever and the LLM
from typing import List  # Needed for lambda function types

# --- 1. Define the System Prompt ---
SYSTEM_PROMPT = """
You are a highly accurate Financial Q&A Assistant. 
Your goal is to answer the user's question ONLY using the provided context. 
If the answer is not found in the context, you MUST clearly state: 
"I cannot find the answer to that question in the available financial documents."
Answer professionally and clearly.

Context: {context}
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)


# --- 2. Context Formatter Function ---
def format_docs(docs: List) -> str:
    """Converts a list of documents into a single string for the prompt."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# --- 3. Assemble the RAG Chain using LCEL ---
# The chain is designed to keep the raw documents for debugging/citation.
rag_chain = (
    # CORRECT LCEL: Start with RunnablePassthrough.assign to create the input dictionary
    RunnablePassthrough.assign(
        # Store the raw documents using the retriever function
        context=RunnableLambda(retriever),
        question=RunnablePassthrough(),
    ).assign(
        # Run the RAG process to get the final answer
        answer=(
            {
                # Format docs for the prompt (using the stored raw context)
                "context": lambda x: format_docs(x["context"]),
                "question": lambda x: x["question"],
            }
            | prompt
            | llm
            | StrOutputParser()
        ),
        # Extract the source filenames for the API response (debugging/citation)
        source_documents=lambda x: [
            doc.metadata.get("source", "Unknown") for doc in x["context"]
        ],
    )
    # Final output selects only the keys needed for the API response
    .pick(["answer", "source_documents"])
)

print("RAG Chain initialized successfully.")

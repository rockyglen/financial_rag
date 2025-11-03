# chatbot_app.py (Frontend UI - Save this outside the FastAPI project folder)

import streamlit as st
import requests
import os
from typing import List

# --- CONFIGURATION ---
# !!! REPLACE THIS with your EC2's public IP address or domain !!!
# Note: You will manually update this file with the EC2's IP after deployment.
API_URL = "http://YOUR_EC2_PUBLIC_IP/predict"

st.set_page_config(page_title="Financial RAG Chatbot")
st.title("Financial Filings Q&A Assistant")
st.markdown("Connected to Deployed FastAPI RAG Service.")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to send data to the deployed FastAPI backend
def ask_rag_backend(prompt: str) -> dict:
    """Sends the user question to the external FastAPI RAG API."""
    payload = {"question": prompt}

    with st.spinner("Searching and synthesizing answer..."):
        try:
            # Send data to the deployed EC2 instance
            response = requests.post(API_URL, json=payload, timeout=90)

            if response.status_code == 200:
                return response.json()
            else:
                st.error(
                    f"Backend API Error: Status {response.status_code}. Check EC2 logs."
                )
                return {
                    "answer": "Error connecting to the RAG backend.",
                    "source_documents": [],
                }

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Check the EC2 IP and Nginx status.")
            return {"answer": "Connection Error.", "source_documents": []}
        except Exception as e:
            st.error(f"An unexpected client error occurred: {e}")
            return {"answer": "An unexpected error occurred.", "source_documents": []}


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Type your financial question here..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get response from RAG backend
    with st.chat_message("assistant"):
        backend_response = ask_rag_backend(prompt)

        answer = backend_response.get("answer", "No answer provided.")
        sources = backend_response.get("source_documents", [])

        # 3. Display answer
        st.markdown(answer)

        # 4. Display sources (Citation)
        if sources:
            # Deduplicate and clean source paths for cleaner display
            unique_sources = sorted(list(set([os.path.basename(s) for s in sources])))
            st.caption("🔍 **Context Sources:** " + ", ".join(unique_sources))

        # 5. Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

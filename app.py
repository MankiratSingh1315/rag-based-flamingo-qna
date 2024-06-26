import streamlit as st
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
message_placeholder = st.empty()

chunks = pickle.load(open('chunks.pkl', 'rb'))

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

OLLAMA_API_URL = "http://localhost:11434/api/generate"
index = faiss.read_index('chunks.index')

def retrieve_chunks(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), top_k)

    retrieved_chunks = []

    for idx in I[0]:
        chunk_text = chunks[idx][0]
        page_number = chunks[idx][1][0]['page_number']

        # Check if the chunk contains undesired content (e.g., questions)
        if len(chunk_text) >= 60 and 'think as you read' not in chunk_text.lower() and 'answer' not in chunk_text.lower() and 'understanding the text' not in chunk_text.lower() and 'talking about the text' not in chunk_text.lower() and 'working with words' not in chunk_text.lower() and 'noticing form' not in chunk_text.lower() and 'things to do' not in chunk_text.lower():
            retrieved_chunks.append((chunk_text, page_number))

    return retrieved_chunks

def generate_response(query, ret):
    context = ""
    for text, page_number in ret:
        context += f"{text}\n"
    prompt = f"""You are a bot to answer the Questions from class 12 NCERT textbooks
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {query}
"""
    data = {
        "model": "llama3",
        "prompt": prompt
    }

    s = requests.Session()

    with s.post(OLLAMA_API_URL, headers=None, data=json.dumps(data), stream=True) as resp:
        response = ""
        for line in resp.iter_lines():
            if line:
                response_line = json.loads(line)['response']
                response += response_line + "▌"
                message_placeholder.markdown(response)
                response = response.rstrip("▌")
        return response

def main():
    st.title("Llama Rag App")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hello! How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            ret = retrieve_chunks(prompt, model, index, chunks)
            full_response = generate_response(prompt, ret)
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

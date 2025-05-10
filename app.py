import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
import google.generativeai as genai
import uuid
import shutil
from datetime import datetime, timedelta
import PyPDF2 # Added for PDF processing

# --- Constants ---
SESSION_DATA_DIR = "session_data"
UPLOAD_DIR_NAME = "uploads"
CHUNKS_FILE_NAME = "chunks.pkl"
INDEX_FILE_NAME = "chunks.index"
STATUS_FILE_NAME = "status.txt"
MAX_FILES = 20
MAX_TOTAL_SIZE_MB = 1024  # 1 GB
SESSION_EXPIRY_HOURS = 2

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel()
else:
    # This will be caught by st.error in the UI if gemini_model is needed
    pass


# --- Helper Functions ---
def get_session_path(session_id):
    return os.path.join(SESSION_DATA_DIR, str(session_id))

def get_session_upload_path(session_id):
    return os.path.join(get_session_path(session_id), UPLOAD_DIR_NAME)

def ensure_session_dirs(session_id):
    session_path = get_session_path(session_id)
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(get_session_upload_path(session_id), exist_ok=True)
    # Create a timestamp file for expiry
    with open(os.path.join(session_path, "timestamp.txt"), "w") as f:
        f.write(datetime.now().isoformat())

def get_session_status(session_id):
    session_path = get_session_path(session_id)
    status_file = os.path.join(session_path, STATUS_FILE_NAME)
    timestamp_file = os.path.join(session_path, "timestamp.txt")

    if not os.path.exists(session_path) or not os.path.exists(timestamp_file):
        return "not_found"

    with open(timestamp_file, "r") as f:
        created_time_str = f.read()
    created_time = datetime.fromisoformat(created_time_str)
    if datetime.now() > created_time + timedelta(hours=SESSION_EXPIRY_HOURS):
        shutil.rmtree(session_path) # Cleanup expired session
        return "expired" # Mark as expired, actual cleanup can be a separate process

    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            return f.read().strip()
    return "pending_processing" # Default if status file not yet created but session exists

def update_session_status(session_id, status):
    session_path = get_session_path(session_id)
    if not os.path.exists(session_path): # Ensure session dir exists before updating status
        ensure_session_dirs(session_id)
    with open(os.path.join(session_path, STATUS_FILE_NAME), "w") as f:
        f.write(status)

def save_uploaded_files(session_id, uploaded_files):
    upload_path = get_session_upload_path(session_id)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return [os.path.join(upload_path, f.name) for f in uploaded_files]

# --- PDF Processing (adapted from notebook) ---
def extract_text_from_pdf_in_chunks(pdf_path, char_limit=1000): # Increased char_limit for better context
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        chunks_data = []
        current_chunk_text = ""
        current_chunk_metadata = [] # Store page numbers for context
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if not page_text: # Skip empty pages
                continue

            # Simple split by paragraphs or force split if too long
            page_sentences = page_text.replace('\\n', ' ').split('. ') 
            
            for sentence in page_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence += "." # Re-add period

                if len(current_chunk_text) + len(sentence) <= char_limit:
                    current_chunk_text += " " + sentence
                    if not current_chunk_metadata or current_chunk_metadata[-1]['page_number'] != page_num + 1:
                         current_chunk_metadata.append({'page_number': page_num + 1})
                else:
                    if current_chunk_text: # Add previous chunk
                        chunks_data.append((current_chunk_text.strip(), [{'page_number': meta['page_number']} for meta in current_chunk_metadata]))
                    current_chunk_text = sentence # Start new chunk
                    current_chunk_metadata = [{'page_number': page_num + 1}]

        if current_chunk_text: # Add the last chunk
            chunks_data.append((current_chunk_text.strip(), [{'page_number': meta['page_number']} for meta in current_chunk_metadata]))
    return chunks_data


def process_pdfs_for_session(session_id, sentence_transformer_model):
    update_session_status(session_id, "processing")
    session_upload_path = get_session_upload_path(session_id)
    pdf_files = [os.path.join(session_upload_path, f) for f in os.listdir(session_upload_path) if f.endswith(".pdf")]

    all_chunks = []
    for pdf_file in pdf_files:
        try:
            all_chunks.extend(extract_text_from_pdf_in_chunks(pdf_file))
        except Exception as e:
            st.error(f"Error processing {pdf_file}: {e}")
            update_session_status(session_id, f"error_processing_{os.path.basename(pdf_file)}")
            return

    if not all_chunks:
        st.warning("No text could be extracted from the provided PDFs.")
        update_session_status(session_id, "error_no_text_extracted")
        return

    chunk_texts = [chunk[0] for chunk in all_chunks]
    
    try:
        chunk_embeddings = sentence_transformer_model.encode(chunk_texts)
        chunk_embeddings_float32 = np.array(chunk_embeddings, dtype=np.float32)

        index = faiss.IndexFlatL2(chunk_embeddings_float32.shape[1])
        index.add(chunk_embeddings_float32)

        session_path = get_session_path(session_id)
        faiss.write_index(index, os.path.join(session_path, INDEX_FILE_NAME))
        with open(os.path.join(session_path, CHUNKS_FILE_NAME), 'wb') as f:
            pickle.dump(all_chunks, f)
        
        update_session_status(session_id, "completed")
    except Exception as e:
        st.error(f"Error during vectorization or saving data: {e}")
        update_session_status(session_id, "error_vectorization")


# --- RAG Components (adapted) ---
# Load a pre-trained SentenceTransformer model (globally or pass as needed)
# Moved model loading to main to avoid reloading on every interaction if possible
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2') 

def retrieve_chunks_session(query, sentence_model, session_id, top_k=5):
    session_path = get_session_path(session_id)
    index_path = os.path.join(session_path, INDEX_FILE_NAME)
    chunks_path = os.path.join(session_path, CHUNKS_FILE_NAME)

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        st.error("Session data not found or incomplete for retrieval.")
        return []

    index = faiss.read_index(index_path)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    query_embedding = sentence_model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), top_k)

    retrieved_chunks_data = []
    for idx in I[0]:
        if 0 <= idx < len(chunks):
            chunk_text = chunks[idx][0]
            # page_numbers = [meta['page_number'] for meta in chunks[idx][1]]
            # Simplified metadata for now, can be expanded
            page_info = f"Source (Page {chunks[idx][1][0]['page_number'] if chunks[idx][1] else 'N/A'})" 
            
            # Basic filter (can be expanded)
            if len(chunk_text) >= 20: # Filter very short/irrelevant chunks
                 retrieved_chunks_data.append((chunk_text, page_info))
    return retrieved_chunks_data

def generate_response_session(query, retrieved_chunks_data, placeholder):
    if not gemini_model:
        st.error("Gemini model is not configured. Please set GEMINI_API_KEY.")
        return "Error: Gemini model not available."

    context = ""
    for text, page_info in retrieved_chunks_data:
        context += f"{text} ({page_info})\\n\\n"
    
    full_prompt = f"""You are an AI assistant. Answer the question based ONLY on the following context from uploaded documents:

{context}

---

Question: {query}
Answer:"""
    
    try:
        response_stream = gemini_model.generate_content(full_prompt, stream=True)
        full_response_text = ""
        for chunk_resp in response_stream:
            if chunk_resp.text:
                full_response_text += chunk_resp.text
                placeholder.markdown(full_response_text + "▌")
        placeholder.markdown(full_response_text)
        return full_response_text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "Sorry, I couldn't generate a response due to an API error."

# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="DocuChat", layout="wide")
    st.title("📄 DocuChat: Chat with your PDFs")

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY environment variable not set. The application cannot function without it.")
        st.stop()
    
    # Load SentenceTransformer model once
    if 'sentence_model' not in st.session_state:
        with st.spinner("Loading sentence model..."):
            st.session_state.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


    if 'current_view' not in st.session_state:
        st.session_state.current_view = "home"
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.processing_done = False


    # --- Home View: Upload or Enter Session ID ---
    if st.session_state.current_view == "home":
        st.header("Welcome!")
        action = st.radio("Choose an action:", ("Upload new PDFs", "Use existing Session ID"))

        if action == "Upload new PDFs":
            st.subheader("Upload your PDF documents")
            uploaded_files = st.file_uploader("Select PDF files (max 20, total 1GB)", type="pdf", accept_multiple_files=True)

            if uploaded_files:
                total_size = sum(f.size for f in uploaded_files)
                if len(uploaded_files) > MAX_FILES:
                    st.error(f"You can upload a maximum of {MAX_FILES} files.")
                elif total_size > MAX_TOTAL_SIZE_MB * 1024 * 1024:
                    st.error(f"Total file size exceeds {MAX_TOTAL_SIZE_MB}MB.")
                else:
                    if st.button("Process Uploaded PDFs"):
                        with st.spinner("Setting up session and saving files..."):
                            session_id = str(uuid.uuid4())
                            st.session_state.session_id = session_id
                            ensure_session_dirs(session_id)
                            save_uploaded_files(session_id, uploaded_files)
                            update_session_status(session_id, "pending_processing")
                            st.session_state.current_view = "processing"
                            st.session_state.processing_done = False # Reset flag
                            st.rerun()
        
        elif action == "Use existing Session ID":
            st.subheader("Enter your Session ID")
            input_session_id = st.text_input("Session ID:")
            if st.button("Load Session"):
                if input_session_id:
                    status = get_session_status(input_session_id)
                    if status == "completed":
                        st.session_state.session_id = input_session_id
                        st.session_state.current_view = "chat"
                        st.session_state.messages = [] # Reset messages for new session
                        st.session_state.processing_done = True
                        st.rerun()
                    elif status == "processing" or status == "pending_processing":
                        st.session_state.session_id = input_session_id
                        st.session_state.current_view = "processing"
                        st.session_state.processing_done = False
                        st.rerun()
                    elif status == "expired":
                        st.error("This session has expired. Please upload your PDFs again.")
                    elif status.startswith("error_"):
                        st.error(f"This session encountered an error: {status.replace('_', ' ')}. Please try uploading again.")
                    else: # not_found
                        st.error("Session ID not found. Please check the ID or upload your PDFs.")
                else:
                    st.warning("Please enter a Session ID.")

    # --- Processing View ---
    elif st.session_state.current_view == "processing":
        session_id = st.session_state.session_id
        st.header(f"Processing Session: {session_id}")
        st.write("Your PDF documents are being processed. This may take a few minutes depending on the size and number of documents.")
        st.info(f"You can save this Session ID: **{session_id}** and come back later if you close this window.")
        
        # Trigger processing if not already done for this view load
        # This is a simplified way; for true background, external workers are needed.
        current_status = get_session_status(session_id)
        if current_status == "pending_processing" and not st.session_state.get('processing_started_for_session', False):
            st.session_state.processing_started_for_session = True # Flag to prevent re-triggering in same run
            with st.spinner("Chunking and vectorizing PDFs... Please wait."):
                process_pdfs_for_session(session_id, st.session_state.sentence_model)
            st.session_state.processing_started_for_session = False # Reset for potential future needs
            st.rerun() # Rerun to check status and move to chat if completed

        elif current_status == "completed":
            st.success("Processing complete!")
            st.session_state.current_view = "chat"
            st.session_state.messages = [] # Reset messages
            st.session_state.processing_done = True
            st.rerun()
        elif current_status.startswith("error_"):
            st.error(f"An error occurred during processing: {current_status.replace('_', ' ')}. Please try creating a new session.")
            if st.button("Go to Home"):
                st.session_state.current_view = "home"
                st.rerun()
        else: # Still processing or another status
            # Add a button to manually refresh status or go home
            if st.button("Refresh Status"):
                st.rerun()
            if st.button("Go to Home (Processing will continue if started)"):
                st.session_state.current_view = "home"
                # Don't reset session_id here, so user can potentially come back
                st.rerun()


    # --- Chat View ---
    elif st.session_state.current_view == "chat":
        if not st.session_state.session_id or not st.session_state.processing_done:
            st.warning("No active session or processing not complete. Redirecting to home.")
            st.session_state.current_view = "home"
            st.rerun()
            st.stop()

        st.header(f"Chat with Documents (Session: {st.session_state.session_id})")

        if st.button("End Session & Start New"):
            # Conceptual: Add cleanup for st.session_state.session_id data here if desired
            # For now, just resets the UI state
            st.session_state.current_view = "home"
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.processing_done = False
            st.rerun()


        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Retrieving relevant information and generating answer..."):
                    retrieved = retrieve_chunks_session(prompt, st.session_state.sentence_model, st.session_state.session_id)
                    if not retrieved:
                        message_placeholder.markdown("I couldn't find relevant information in your documents to answer that question.")
                        st.session_state.messages.append({"role": "assistant", "content": "I couldn't find relevant information in your documents to answer that question."})
                    else:
                        full_response = generate_response_session(prompt, retrieved, message_placeholder)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
    


if __name__ == "__main__":
    os.makedirs(SESSION_DATA_DIR, exist_ok=True)
    main()

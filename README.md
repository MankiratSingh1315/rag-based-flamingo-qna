# PDF RAG Q&A with Gemini and Streamlit

This project implements a Retrieval Augmented Generation (RAG) based Question & Answering system. Users can upload PDF documents, which are then processed and stored locally for a session. The system uses Google's Gemini API to answer questions based on the content of the uploaded PDFs. The user interface is built with Streamlit.

## How it Works

1.  **User Interaction (Streamlit UI):**
    *   **New Session:** Users can upload one or more PDF files (up to 20 files, 1GB total size limit).
    *   **Existing Session:** Users can enter a previously generated Session ID to continue an existing session.

2.  **Session Management (Local File Storage):**
    *   When new PDFs are uploaded, a unique Session ID is generated.
    *   All data related to a session (uploaded PDFs, extracted text chunks, FAISS vector index, status, and timestamp) is stored in a local directory named `session_data/<session_id>/`.
    *   Sessions automatically expire after 2 hours. Expired session data is periodically cleaned up.

3.  **PDF Processing (Background - Simulated in Streamlit):**
    *   **Text Extraction:** Text is extracted from the uploaded PDFs.
    *   **Chunking:** The extracted text is divided into smaller, manageable chunks.
    *   **Vectorization:** Each chunk is converted into a numerical vector (embedding) using a SentenceTransformer model (`all-MiniLM-L6-v2`).
    *   **Indexing:** The embeddings are stored in a FAISS index for efficient similarity searching.

4.  **RAG Pipeline for Q&A:**
    *   When a user asks a question:
        *   The question is converted into an embedding.
        *   The FAISS index is searched to find the most relevant text chunks from the uploaded PDFs.
        *   These retrieved chunks, along with the original question, are provided as context to the Gemini Pro model.
        *   Gemini generates an answer based *only* on the provided context.
    *   The answer is streamed back to the user in the chat interface.

## Features

*   Upload multiple PDF documents.
*   Session-based data management with local file storage.
*   Automatic session expiry and cleanup.
*   Retrieval Augmented Generation using Gemini Pro.
*   Streaming responses in the chat interface.
*   User-friendly UI built with Streamlit.
*   Includes a Jupyter Notebook (`llama_rag.ipynb`) for an alternative, more granular way to interact with the components.

## Prerequisites

*   Python 3.8 or higher
*   Access to Google's Gemini API and a `GEMINI_API_KEY`.

## Setup and Running the Application

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd rag-based-flamingo-qna
    ```

2.  **Create and Activate a Python Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *On Windows, use `.\.venv\Scripts\activate`*

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application:**
    ```bash
    GEMINI_API_KEY=api-key streamlit run app.py
    ```
    This will open the application in your web browser.

5.  **Using the Application:**
    *   **Option 1: Upload New PDFs:**
        *   Select "Upload new PDFs".
        *   Use the file uploader to select your PDF documents.
        *   Click "Process Uploaded PDFs".
        *   A Session ID will be displayed. **Save this ID** if you want to resume the session later.
        *   Wait for the processing to complete. The UI will indicate when it's ready.
        *   Once processed, the chat interface will appear, allowing you to ask questions about your documents.
    *   **Option 2: Use Existing Session ID:**
        *   Select "Use existing Session ID".
        *   Enter your previously saved Session ID.
        *   Click "Load Session".
        *   If the session is valid and processed, the chat interface will appear.

## Jupyter Notebook (`llama_rag.ipynb`)

The repository also includes `llama_rag.ipynb`, which provides a step-by-step walkthrough of the PDF processing, RAG pipeline, and session management. This can be useful for understanding the individual components or for debugging.

To use the notebook:
1.  Ensure you have Jupyter Notebook or JupyterLab installed (`pip install notebook` or `pip install jupyterlab`).
2.  Run `jupyter notebook` or `jupyter lab` from the project directory.
3.  Open `llama_rag.ipynb` and run the cells.

## Project Structure

```
.streamlit/
  config.toml       # Streamlit configuration (e.g., for file watcher)
app.py              # Main Streamlit application file
llama_rag.ipynb     # Jupyter Notebook for testing and exploration
README.md           # This file
requirements.txt    # Python dependencies
flamingo-class-12/  # Example PDF files (can be replaced or removed)
  ch1.pdf
  ...
session_data/       # Directory for storing session-specific data (created automatically)
  <session_id>/     # Data for a specific session
    chunks.index
    chunks.pkl
    status.txt
    timestamp.txt
    uploads/
      <uploaded_pdf_name>.pdf
```

## Notes

*   The PDF processing (chunking and vectorization) happens within the Streamlit script. For very large files or a high number of concurrent users, this might be slow. In a production environment, this would typically be offloaded to a separate worker process and task queue.
*   Session data is stored locally. This is suitable for single-user or development setups. For a multi-user or production application, a more robust storage solution (like a database or cloud storage) would be needed for session management and data persistence.
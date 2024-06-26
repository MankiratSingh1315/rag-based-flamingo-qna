# Flamingo Class 12 Q/A Bot

## Project Overview
This project is a RAG based Streamlit web application designed to answer class 12 Book flamingo Q/A. 

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- llama3 model API served by Ollama
- Python 3.7 or higher
- pip (Python package installer)
- Jupyter Notebook (for generating the required files)

### Installation
1. **Clone the repository:**
    ```sh
    git clone https://github.com/MankiratSingh1315.git
    cd yourproject
    ```

2. **Install required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Generate `.pkl` and `index` files:**
    Open the Jupyter Notebook (`generate_files.ipynb`) and run all cells to generate the necessary `.pkl` and `index` files.

    ```sh
    jupyter notebook generate_files.ipynb
    ```

### Running the Application
1. **Start the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

2. **Access the application:**
    Open your web browser and navigate to `http://localhost:8501`.
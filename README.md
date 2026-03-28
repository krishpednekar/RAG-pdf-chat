# Chat with Multiple PDFs using Retrieval-Augmented Generation (RAG)

This project is an interactive web application that allows users to upload multiple PDF documents and ask questions about their content using a large language model (LLM).

---

## Features

* Upload and process multiple PDF files
* Extract and analyze document text
* Efficient text chunking for improved retrieval
* Vector search using FAISS
* Conversational question-answering with memory
* Web interface built with Streamlit
* Local inference using Ollama

---

## Tech Stack

* Frontend/UI: Streamlit
* LLM: Ollama (LLaMA 3)
* Embeddings: Sentence Transformers
* Vector Store: FAISS
* Framework: LangChain
* PDF Processing: PyPDF2

---

## Project Structure

```
├── app.py                # Main application
├── htmlTemplates.py      # UI templates
├── requirements.txt      # Dependencies
├── .env                  # Environment variables (optional)
└── README.md             # Documentation
```

---

## Installation and Setup

### 1. Clone the repository

```
git clone https://github.com/krishpednekar/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

---

### 2. Create a virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Install Ollama

Download and install from:
https://ollama.com

Then pull the model:

```
ollama run llama3
```

---

### 5. Run the application

```
python -m streamlit run app.py
```

---

## How It Works

1. PDF documents are uploaded through the interface
2. Text is extracted using PyPDF2
3. The text is split into manageable chunks
4. Embeddings are generated using Sentence Transformers
5. Data is stored in a FAISS vector database
6. Relevant chunks are retrieved based on user queries
7. The LLM generates a contextual response

---

## Use Cases

* Study assistance for notes and textbooks
* Document analysis for reports and research papers
* Business document question-answering
* Legal or policy document exploration

---

## Notes

* Ensure Ollama is running before starting the application
* Large PDF files may take longer to process
* Initial model download may be several gigabytes

---

## Future Improvements

* Add source citations for responses
* Enable streaming responses
* Deploy to a cloud platform
* Improve user interface and experience

---

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.

---

## Author

Krish Pednekar
https://github.com/krishpednekar

# Chat with your PDF

A RAG-based (Retrieval Augmented Generation) chatbot that lets you upload a PDF and ask questions about it in plain English. Instead of the AI guessing or making things up, it pulls answers directly from the content of your document.

## How it works

1. You upload a PDF
2. The app breaks the PDF into small chunks of text
3. Each chunk is converted into embeddings (numerical representations of meaning) using Gemini's embedding model
4. These embeddings are stored in a FAISS vector database for fast searching
5. When you ask a question, the app finds the most relevant chunks and passes them to Gemini
6. Gemini generates an answer based only on that retrieved content

## Tech stack

- **Python** – core logic
- **Streamlit** – web interface
- **LangChain** – connects the PDF loading, chunking, and retrieval pipeline
- **FAISS** – vector database for similarity search
- **Google Gemini API** – embeddings and answer generation

## Setup instructions

1. Clone this repository
```bash
git clone https://github.com/aishwarya-analyst/chat-with-pdf-rag.git
cd chat-with-pdf-rag
```

2. Install the required packages
```bash
pip install streamlit langchain langchain-community langchain-google-genai langchain-classic langchain-text-splitters faiss-cpu pypdf python-dotenv
```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

4. Open `app.py` and replace the placeholder with your own API key:
```python
os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_KEY_HERE"
```

5. Run the app
```bash
streamlit run app.py
```

6. Upload a PDF and start asking questions

## Notes

- Do not commit your real API key to GitHub. Keep it as a placeholder in the code and set it locally or through environment variables when running.
- Model names for embeddings and chat may change over time as Google updates the Gemini API. If you hit a "model not found" error, check the [Gemini API docs](https://ai.google.dev/gemini-api/docs/models) for the current model names.

## Author

Aishwarya AR

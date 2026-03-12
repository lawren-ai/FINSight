# FINSight — LLM Financial Analyst

A RAG-powered API that answers questions from financial documents using OpenAI and FAISS.

## How it works
1. Run `ingest.py` to extract, chunk and embed your PDFs into a FAISS vector store
2. Run `main.py` to start the FastAPI server
3. POST a question to `/query/` and get an answer grounded in your documents

## Stack
- FastAPI
- LangChain
- OpenAI (text-embedding-3-small + gpt-4o-mini)
- FAISS
- PyPDF2

## Setup
1. Clone the repo
2. Create a `.env` file with your `OPENAI_API_KEY`
3. Install dependencies: `pip install -r requirements.txt`
4. Run `python ingest.py`
5. Run `uvicorn main:app --reload`
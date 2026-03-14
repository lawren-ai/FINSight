# FINSight — LLM Financial Analyst

An AI-powered RAG (Retrieval-Augmented Generation) system that answers natural language 
questions about company financial filings, powered by OpenAI and FAISS.

## What it does
- Fetches 10-K filings directly from the SEC EDGAR API
- Chunks, embeds and indexes the documents into a FAISS vector store
- Answers natural language questions grounded strictly in the filing content
- Evaluated using RAGAS — faithfulness, answer relevancy, context recall, context precision

## Evaluation Results (Apple 10-K, FY2025)
| Question | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|---|
| Total revenue FY2025 | 1.0 | 1.0 | 1.0 | 0.83 |
| Main product categories | 1.0 | 0.95 | 1.0 | 1.0 |

## Stack
- FastAPI — REST API
- LangChain — RAG orchestration
- OpenAI — embeddings (text-embedding-3-small) + LLM (gpt-4o-mini)
- FAISS — vector store
- RAGAS — RAG evaluation framework
- SEC EDGAR API — live financial data ingestion
- BeautifulSoup — HTML parsing

## Project Structure
```
FINSight/
├── data/              # local PDFs (gitignored)
├── faiss_index/       # generated vector store (gitignored)
├── ingest.py          # fetches SEC filings, chunks, embeds and saves index
├── main.py            # FastAPI app — POST /query/ endpoint
├── evaluate.py        # RAGAS evaluation pipeline
└── .env               # API keys (gitignored)
```

## Setup
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with: `OPENAI_API_KEY=your-key-here`
6. Run ingestion: `python ingest.py`
7. Start the API: `uvicorn main:app --reload`

## Usage
Send a POST request to `http://localhost:8000/query/`:
```json
{
  "query": "What was Apple's total revenue for fiscal year 2025?"
}
```

Response:
```json
"Apple's total revenue for fiscal year 2025 was $416,161 million."
```

## Known Limitations
- Vocabulary mismatch: queries using informal terms (e.g. "CEO") may not retrieve 
  chunks that use formal language ("Chief Executive Officer"). Query expansion is 
  a planned improvement.
- Currently hardcoded to Apple (AAPL). Multi-ticker support is in development.

## Roadmap
- [ ] Accept any company ticker as input
- [ ] Query expansion to handle vocabulary mismatch
- [ ] Streamlit frontend
- [ ] Docker deployment
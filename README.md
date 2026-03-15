# FINSight — LLM Financial Analyst

An AI-powered RAG (Retrieval-Augmented Generation) system that answers natural language
questions about any publicly traded company's financial filings, powered by OpenAI and FAISS.

## What it does
- Accepts any stock ticker (AAPL, MSFT, TSLA, etc.)
- Fetches 10-K filings directly from the SEC EDGAR API — no manual downloads
- Chunks, embeds and indexes the documents into a FAISS vector store
- Answers natural language questions grounded strictly in the filing content
- Evaluated using RAGAS — faithfulness, answer relevancy, context recall, context precision

## Evaluation Results
| Question | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|---|
| What was Apple's total revenue for FY2025? | 1.0 | 1.0 | 1.0 | 0.83 |
| What are Apple's main product categories? | 1.0 | 0.95 | 1.0 | 1.0 |
| What was Microsoft's total revenue for FY2025? | 1.0 | 1.0 | 1.0 | 1.0 |

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
├── faiss_index/       # generated vector store (gitignored)
├── ingest.py          # fetches SEC filings, chunks, embeds and saves index
├── main.py            # FastAPI app — POST /query/ endpoint
├── evaluate.py        # RAGAS evaluation pipeline
├── requirements.txt   
└── .env               # API keys (gitignored)
```

## Setup
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with: `OPENAI_API_KEY=your-key-here`

## Usage

### Ingest a company's filings
```bash
# Ingest the 3 most recent 10-K filings for any ticker
python ingest.py AAPL 3
python ingest.py MSFT 2
python ingest.py TSLA 5
```

### Start the API
```bash
uvicorn main:app --reload
```

### Query the API
Send a POST request to `http://localhost:8000/query/`:
```json
{
  "query": "What was Microsoft's total revenue for fiscal year 2025?"
}
```

Response:
```json
"Microsoft's total revenue for fiscal year 2025 was $281,724 million."
```

### Run evaluation
```bash
python evaluate.py
```

## Known Limitations
- Vocabulary mismatch: queries using informal terms (e.g. "CEO") may not retrieve
  chunks that use formal language ("Chief Executive Officer"). Query expansion is
  a planned improvement.
- Large filings (200+ pages) may take 1-2 minutes to ingest due to embedding API calls.
- Evaluation test set is currently manually curated — automated test generation is
  a planned improvement.

## Roadmap
- [ ] Streamlit frontend
- [ ] Docker deployment
- [ ] Query expansion to handle vocabulary mismatch
- [ ] Automated evaluation test set generation
- [ ] Multi-company comparison queries
# FINSight — LLM Financial Analyst

An AI-powered RAG (Retrieval-Augmented Generation) system that answers natural language questions about any publicly traded company's financial filings, powered by OpenAI and FAISS.

## What it does
- **On-Demand Indexing**: Automatically detects new tickers, fetches 10-K filings from the SEC EDGAR API, and builds a local vector store library.
- **Dynamic Context**: Switches between company data seamlessly based on the user-provided ticker.
- **Grounded Answers**: Answers natural language questions grounded strictly in the retrieved filing content to prevent hallucinations.
- **Evaluated with RAGAS**: Performance is measured using faithfulness, answer relevancy, context recall, and context precision.

## Evaluation Results

| Question | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|---|
| What was Apple's total revenue for FY2025? | 1.0 | 1.0 | 1.0 | 0.83 |
| What are Apple's main product categories? | 1.0 | 0.95 | 1.0 | 1.0 |
| What was Microsoft's total revenue for FY2025? | 1.0 | 1.0 | 1.0 | 1.0 |

## Stack
- **FastAPI** — REST API backend
- **Gradio** — Interactive Web Chat Interface
- **LangChain** — RAG orchestration
- **OpenAI** — Embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
- **FAISS** — Vector store for high-performance retrieval
- **RAGAS** — RAG evaluation framework
- **SEC EDGAR API** — Live financial data ingestion
- **BeautifulSoup** — HTML parsing and cleaning

## Project Structure
```text
FINSight/
├── faiss_*/           # ticker-specific vector stores (gitignored)
├── ingest.py          # logic to fetch SEC filings, chunk, and embed
├── app.py             # Gradio Web Interface (UI)
├── main.py            # FastAPI App — POST /query/ endpoint
├── evaluate.py        # RAGAS evaluation pipeline
├── requirements.txt   
└── .env               # API keys (gitignored)
```

## Setup
Clone the repo.
Create a virtual environment: `python -m venv venv`
Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux).
Install dependencies: `pip install -r requirements.txt`
Create a `.env` file with: `OPENAI_API_KEY=your-key-here`

## Usage
1. Start the Web Interface (UI)
```bash
python app.py
```

Enter a ticker (e.g., NVDA) and ask a question. If the data isn't local, the app will automatically ingest it.

2. Start the API
```bash
uvicorn main:app --reload
```

3. Query the API
Send a POST request to `http://localhost:8000/query/`:
```json
{
  "ticker": "AAPL",
  "query": "What was Apple's total revenue for fiscal year 2025?"
}
```

4. Run Evaluation
```bash
python evaluate.py
```


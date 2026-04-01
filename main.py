import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from ingest import ingest_company

class UserQuery(BaseModel):
    query: str
    ticker: str

app = FastAPI()

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o-mini")

@app.post("/query/")
def query_rag(user_query: UserQuery):
    question = user_query.query
    ticker = user_query.ticker.upper().strip()
    folder_name = f"faiss_{ticker}"
    if os.path.exists(f"faiss_{ticker}"):
        vector_store = FAISS.load_local(folder_name, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Ingesting data for {ticker}...")
        vector_store = ingest_company(ticker, num_filings=3)
        if not vector_store:
            raise HTTPException(status_code=404, detail=f"Could not find SEC data for {ticker}")
    
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([docs.page_content for docs in docs])

    if not context.strip():
        return {"answer": "I don't know (no relevant context found).", "ticker": ticker}
    messages = [
        ("system", "You are a financial analyst assistant. Answer questions based only on the provided context. If the answer is not in the context, say you don't know."),
        ("human", f"Context:\n{context}\n\nQuestion: {question}")
    ]
    ai_msg = llm.invoke(messages)

    return {"answer": ai_msg.content, "ticker": ticker}

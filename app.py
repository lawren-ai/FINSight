import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import gradio as gr
from ingest import ingest_company

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")


def query_rag(question, history, ticker):
    if not ticker:
        return "Please enter a ticker symbol first. (e.g., MSFT)"
    ticker = ticker.upper()
    folder_name = f"faiss_{ticker}"
    if os.path.exists(folder_name):
        vector_store = FAISS.load_local(folder_name, embeddings, allow_dangerous_deserialization=True)
    else:
        gr.Info(f"New ticker {ticker} detected. Downloading filings...")
        vector_store = ingest_company(ticker, num_filings=3)
        if not vector_store:
            return "Failed to retrieve data for that ticker."
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    messages = [
        ("system", "You are a knowledeable financial analyst assistant for {ticker}. Answer questions based only on the provided context. If the answer is not in the context, say you don't know."),
        ("human", f"Context:\n{context}\n\nQuestion: {question}")
    ]
    ai_msg = llm.invoke(messages)

    return ai_msg.content

with gr.Blocks() as demo:
    gr.Markdown("# FINSight: Financial Document Q&A")
    ticker_input = gr.Textbox(label="Enter Ticker Symbol (e.g., AAPL)", placeholder="AAPL")
    gr.ChatInterface(
        fn=query_rag,
        additional_inputs=[ticker_input]
    )

demo.launch()
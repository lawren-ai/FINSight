from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

class UserQuery(BaseModel):
    query: str

app = FastAPI()

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o-mini")

@app.post("/query/")
def query_rag(query: UserQuery):
    question = query.query
    results = vector_store.similarity_search_with_score(question, k=3)
    context_chunk = []
    for res, score in results:
        context_chunk.append(res.page_content)
        context = "\n\n".join(context_chunk)
    messages = [
    ("system", "You are a financial analyst assistant. Answer questions based only on the provided context. If the answer is not in the context, say you don't know."),
    ("human", f"Context:\n{context}\n\nQuestion: {question}")
    ]
    ai_msg = llm.invoke(messages)

    return ai_msg.content

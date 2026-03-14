from langchain_community.vectorstores import FAISS
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o-mini")

test_questions = [
    {
        "question": "What was Apple's total revenue for fiscal year 2025?",
        "ground_truth": "Apple's total revenue for fiscal year 2025 was $416,161 million."
    },
    {
        "question": "What are Apple's main product categories?",
        "ground_truth": "Apple's main product categories are iPhone, Mac, iPad, Wearables, Home and Accessories, and Services."
    },
    {
        "question": "Who is Apple's Chief Executive Officer?",
        "ground_truth": "Apple's Chief Executive Officer is Tim Cook."
    }
]

client = AsyncOpenAI()
ragas_llm = llm_factory("gpt-4o-mini", client=client)
ragas_embeddings = embedding_factory('openai', model="text-embedding-3-small", client=client, interface='modern')

async def evaluate_sample(question, llm_answer, retrieved_contexts, ground_truth):
    faithfulness = Faithfulness(llm=ragas_llm)
    relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    recall = ContextRecall(llm=ragas_llm)
    precision = ContextPrecision(llm=ragas_llm)

    return {
        "question": question,
        "faithfulness": (await faithfulness.ascore(user_input=question, response=llm_answer, retrieved_contexts=retrieved_contexts)).value,
        "answer_relevancy": (await relevancy.ascore(user_input=question, response=llm_answer)).value,
        "context_recall": (await recall.ascore(user_input=question, retrieved_contexts=retrieved_contexts, reference=ground_truth)).value,
        "context_precision": (await precision.ascore(user_input=question, retrieved_contexts=retrieved_contexts, reference=ground_truth)).value,
    }

async def main():
    all_results = []
    for item in test_questions:
        question = item['question']
        ground_truth = item['ground_truth']

        search_results = vector_store.similarity_search_with_score(question, k=5)
        retrieved_contexts = [res.page_content for res, score in search_results]
        context_text = "\n\n".join(retrieved_contexts)

        messages = [
            ("system", "You are a financial analyst assistant. Answer questions based only on the provided context. If the answer is not in the context, say you don't know."),
            ("human", f"Context:\n{context_text}\n\nQuestion: {question}")
        ]
        ai_msg = llm.invoke(messages)
        llm_answer = ai_msg.content

        print(f"Question: {question}")
        print(f"Answer: {llm_answer}")
        print("---")

        result = await evaluate_sample(question, llm_answer, retrieved_contexts, ground_truth)
        all_results.append(result)
        print(result)

    return all_results

asyncio.run(main())
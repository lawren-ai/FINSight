import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

file_directory = "C:/Users/Ayotunde/Desktop/FINSight/data"

full_text = ""
texts = []
for filename in os.listdir(file_directory):
    filename = os.path.join(file_directory, filename)
    print(f"Extracting texts from {filename}")
    reader =  PdfReader(f"{filename}")
    for page in reader.pages:
        full_text += page.extract_text() or ""
    full_text = full_text.replace("\n", " ")
    if full_text != "":
        print(f"Texts fully extracted for {filename}")
    else:
        print("There are no texts to be extracted")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts += text_splitter.split_text(full_text)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_texts(texts, embedding=embeddings)
vector_store.save_local("faiss_index")

print(f"Vector store created and saved, Total chunks: {len(texts)}")













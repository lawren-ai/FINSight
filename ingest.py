import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings



import requests
import json
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


import requests
headers = {
    "User-Agent": "FINSight gotskin4@gmail.com"
}
def get_cik_from_ticker(ticker_symbol):
    headers = {'User-Agent': 'FINSight gotskin4@gmail.com'}
    url = "https://www.sec.gov/files/company_tickers.json"
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    for entry in data.values():
        if entry['ticker'].upper() == ticker_symbol.upper():
            return str(entry['cik_str']).zfill(10)
    
    print(f"CIK not found for ticker: {ticker_symbol}")
    return None

def get_10k_accession_numbers(cik, num_filings=3):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    data = response.json()
    
    filings = data['filings']['recent']
    form_types = filings['form']
    accession_numbers = filings['accessionNumber']
    
    results = []
    for i, form_type in enumerate(form_types):
        if form_type == '10-K':
            results.append(accession_numbers[i])
        if len(results) == num_filings:
            break
    return results



def ingest_company(ticker: str, num_filings: int = 3):
    cik = get_cik_from_ticker(ticker)
    if not cik:
        print(f"Could not find CIK for {ticker}")
        return

    accession_numbers = get_10k_accession_numbers(cik, num_filings)
    all_texts = []

    for accession in accession_numbers:
        print(f"Fetching filing: {accession}")
        accession_formatted = accession.replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_formatted}/{accession}-index.htm"

        response = requests.get(index_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        doc_url = None
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('.htm') and 'exhibit' not in href.lower() and 'ix?doc=' in href:
                clean_path = href.replace('/ix?doc=', '')
                doc_url = f"https://www.sec.gov{clean_path}"
                break

        if not doc_url:
            print(f"Could not find main document for {accession}")
            continue

        doc_response = requests.get(doc_url, headers=headers)
        doc_soup = BeautifulSoup(doc_response.text, 'html.parser')

        for tag in doc_soup(['script', 'style', 'ix:header', 'ix:hidden', 'nav', 'header', 'footer']):
            tag.decompose()

        doc_text = doc_soup.get_text(separator=" ", strip=True)
        doc_text = doc_text.replace("\n", " ")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = text_splitter.split_text(doc_text)
        all_texts.extend(texts)
        print(f"Extracted {len(texts)} chunks from {accession}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(all_texts, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print(f"Done. Total chunks indexed: {len(all_texts)}")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    num_filings = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ingest_company(ticker, num_filings)












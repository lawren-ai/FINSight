import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS




import requests
import json

# # The SEC requires a descriptive User-Agent
headers = {
    "User-Agent": "FINSight gotskin4@gmail.com"
}

# # Example: Get all submissions metadata for Apple (CIK 0000320193)
cik = "0000320193" # CIKs need to be 10 digits with leading zeros for the URL, but the API response has them without
# url = f"https://data.sec.gov/submissions/CIK{cik}.json"

# try:
#     response = requests.get(url, headers=headers)
#     response.raise_for_status() # Raise an exception for bad status codes

#     submissions_data = response.json()
#     print(f"Company name: {submissions_data['name']}")


#     filings = submissions_data['filings']['recent']
#     form_types = filings['form']
#     accession_numbers = filings['accessionNumber']
#     filing_dates = filings['filingDate']


#     for i, form_type in enumerate(form_types):
#         if form_type == '10-K':
#             print(f"Date: {filing_dates[i]}, Accession: {accession_numbers[i]}")

# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

accession = "0000320193-25-000079"
accession_formatted = accession.replace("-", "")
url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_formatted}/{accession}-index.htm"

response = requests.get(url, headers=headers)
print(response.text[:2000])
soup = BeautifulSoup(response.text, 'html.parser')

for link in soup.find_all('a'):
    href = link.get('href', '')
    text = link.text.strip()
    if 'aapl-' in href and href.endswith('.htm') and 'exhibit' not in href.lower():
        # extract the clean path
        clean_path = href.replace('/ix?doc=', '')

        doc_url = f"https://www.sec.gov{clean_path}"

doc_response = requests.get(doc_url, headers=headers)
doc_soup = BeautifulSoup(doc_response.text, 'html.parser')


for tag in doc_soup(['script', 'style', 'ix:header', 'ix:hidden', 'nav', 'header', 'footer']):
    tag.decompose()

doc_text = doc_soup.get_text(separator=" ", strip=True)
doc_text = doc_text.replace("\n", " ")



text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(doc_text)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_texts(texts, embedding=embeddings)
vector_store.save_local("faiss_index")

print(f"Vector store created and saved, Total chunks: {len(doc_text)}")













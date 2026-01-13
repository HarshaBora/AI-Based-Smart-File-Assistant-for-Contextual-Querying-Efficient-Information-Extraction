import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    BSHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_pdf(path):
    return PyPDFLoader(path).load()

def load_docx(path):
    return Docx2txtLoader(path).load()

def load_html(path):
    return BSHTMLLoader(path).load()

def load_txt(path):
    return TextLoader(path, encoding="utf-8").load()

documents = []

documents.extend(load_pdf(os.path.join(DATA_DIR, "the_constitution_of_india.pdf")))
documents.extend(load_docx(os.path.join(DATA_DIR, "THE INDIAN PENAL CODE.docx")))
documents.extend(load_html(os.path.join(DATA_DIR, "India Code_ Section Details.html")))
documents.extend(load_txt(os.path.join(DATA_DIR, "legal document.txt")))

# -----------------------------
# Text splitting

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# -----------------------------
# Save output
output_file = os.path.join(BASE_DIR, "output_chunks.txt")

with open(output_file, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks, start=1):
        f.write(f"--- Chunk {i} ---\n")
        f.write(chunk.page_content)
        f.write("\n\n")

print("✅ Task 1 completed successfully")

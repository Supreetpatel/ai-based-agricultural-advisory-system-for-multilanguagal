import os
import json
import pandas as pd
from pathlib import Path

from pypdf import PdfReader # pyright: ignore[reportMissingImports]
from docx import Document # pyright: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Configuration
# ==========================================

DATA_DIR = "data"
VECTOR_STORE = "vector_store"

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
    ".csv",
    ".xlsx",
    ".md",
}

os.makedirs(VECTOR_STORE, exist_ok=True)


# ==========================================
# Load Embedding Model
# ==========================================

print("Loading Embedding Model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model Loaded Successfully.\n")


# ==========================================
# Chunk Splitter
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)


# ==========================================
# File Readers
# ==========================================

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path):
    pdf = PdfReader(path)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def read_docx(path):
    doc = Document(path)

    return "\n".join(
        para.text
        for para in doc.paragraphs
    )


def read_csv(path):
    df = pd.read_csv(path)

    return "\n".join(
        df.astype(str)
        .apply(lambda row: " ".join(row), axis=1)
    )


def read_xlsx(path):
    df = pd.read_excel(path)

    return "\n".join(
        df.astype(str)
        .apply(lambda row: " ".join(row), axis=1)
    )


def read_md(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================
# Read File Based on Extension
# ==========================================

def read_file(file_path):

    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        return read_txt(file_path)

    elif ext == ".pdf":
        return read_pdf(file_path)

    elif ext == ".docx":
        return read_docx(file_path)

    elif ext == ".csv":
        return read_csv(file_path)

    elif ext == ".xlsx":
        return read_xlsx(file_path)

    elif ext == ".md":
        return read_md(file_path)

    else:
        raise ValueError(f"Unsupported File: {ext}")


# ==========================================
# Get All Files
# ==========================================

def get_all_files():

    files = []

    for file in Path(DATA_DIR).iterdir():

        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file)

    return files


# ==========================================
# Chunk Text
# ==========================================

def chunk_text(text):

    return text_splitter.split_text(text)


# ==========================================
# Generate Embeddings
# ==========================================

def generate_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings.tolist()


# ==========================================
# Save JSON
# ==========================================

def save_json(file_name, chunks, embeddings):

    data = []

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings),
        start=1
    ):

        data.append(
            {
                "id": f"{Path(file_name).stem}_{i}",
                "text": chunk,
                "embedding": embedding,
                "metadata": {
                    "source": file_name,
                    "chunk": i
                }
            }
        )

    output = os.path.join(
        VECTOR_STORE,
        f"{Path(file_name).stem}.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved → {output}")
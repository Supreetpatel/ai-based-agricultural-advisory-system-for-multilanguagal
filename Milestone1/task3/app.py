import os
import json
import traceback
import pandas as pd
from pathlib import Path
from pypdf import PdfReader # pyright: ignore[reportMissingImports]
from docx import Document # pyright: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = "data"         
OUTPUT_DIR = "vector_store"  
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv"}

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def read_csv(file_path):
    df = pd.read_csv(file_path)
    return "\n".join(df.astype(str).apply(lambda row: " ".join(row), axis=1))


READERS = {
    ".txt": read_txt,
    ".md": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".csv": read_csv,
}


def read_file(file_path):
    ext = Path(file_path).suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return reader(file_path)


def discover_files(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    files = [
        str(p) for p in sorted(data_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files

def process_file(file_path, output_dir):
    print(f"Processing: {file_path}")

    text = read_file(file_path)
    if not text or not text.strip():
        print(f"  Skipped (no extractable text): {file_path}")
        return 0

    chunks = text_splitter.split_text(text)
    if not chunks:
        print(f"  Skipped (no chunks produced): {file_path}")
        return 0

    embeddings = model.encode(chunks, convert_to_numpy=True)

    vector_data = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
        vector_data.append(
            {
                "id": idx,
                "source_file": os.path.basename(file_path),
                "text": chunk,
                "embedding": embedding.tolist(),
            }
        )

    out_name = Path(file_path).stem + "_vectors.json"
    out_path = Path(output_dir) / out_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vector_data, f, indent=4, ensure_ascii=False)

    print(f"  Saved {len(vector_data)} chunks -> {out_path}")
    return len(vector_data)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = discover_files(DATA_DIR)
    if not files:
        print(f"No supported files found in '{DATA_DIR}'. "
              f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    print(f"Found {len(files)} file(s) to process.\n")

    total_chunks = 0
    processed = 0
    failed = []

    for file_path in files:
        try:
            total_chunks += process_file(file_path, OUTPUT_DIR)
            processed += 1
        except Exception as e:
            print(f"  ERROR processing {file_path}: {e}")
            traceback.print_exc()
            failed.append(file_path)
        print()

    print("Done!")
    print(f"Files processed : {processed}/{len(files)}")
    print(f"Total chunks    : {total_chunks}")
    print(f"Output folder   : {OUTPUT_DIR}/")
    if failed:
        print(f"Failed files    : {failed}")


if __name__ == "__main__":
    main()
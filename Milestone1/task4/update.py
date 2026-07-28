import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

from utils import (
    read_file,
    chunk_text,
    generate_embeddings,
    save_json
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

if INDEX_NAME is None:
    raise ValueError("INDEX_NAME is not set.")

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

document = input("Enter document name (Example: ai.pdf): ").strip()

path = Path("data") / document

if not path.exists():
    print("Document not found.")
    exit()

print("\nDeleting old vectors...")

index.delete(
    filter={
        "source": document
    }
)

print("Old vectors deleted.")

json_file = Path("vector_store") / f"{path.stem}.json"

if json_file.exists():
    json_file.unlink()
    print("Old JSON removed.")

print("\nReading updated document...")

text = read_file(path)

chunks = chunk_text(text)

embeddings = generate_embeddings(chunks)

save_json(document, chunks, embeddings)

vectors = []

for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings),
        start=1
):

    vectors.append(
        {
            "id": f"{path.stem}_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": document,
                "chunk": i
            }
        }
    )

index.upsert(vectors=vectors)

print("\nDocument Updated Successfully!")
print(f"Total Chunks : {len(vectors)}")
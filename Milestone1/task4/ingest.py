import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

from utils import (
    get_all_files,
    read_file,
    chunk_text,
    generate_embeddings,
    save_json
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME is None:
    raise ValueError("INDEX_NAME environment variable is not set")

index = pc.Index(INDEX_NAME)

print(f"\nConnected to Index: {INDEX_NAME}\n")


files = get_all_files()

if not files:
    print("No supported files found inside data folder.")
    exit()

total_vectors = 0

for file in files:

    print("=" * 60)
    print(f"Processing : {file.name}")

    try:

        text = read_file(file)

        if not text.strip():
            print("Empty file. Skipping...")
            continue

        chunks = chunk_text(text)

        print(f"Chunks Created : {len(chunks)}")

        embeddings = generate_embeddings(chunks)

        save_json(file.name, chunks, embeddings)
        
        vectors = []

        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings),
            start=1
        ):

            vectors.append(
                {
                    "id": f"{Path(file).stem}_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "source": file.name,
                        "chunk": i
                    }
                }
            )

        index.upsert(vectors=vectors)

        print(f"Uploaded {len(vectors)} vectors")

        total_vectors += len(vectors)

    except Exception as e:

        print(f"Error Processing {file.name}")
        print(e)

print("\n" + "=" * 60)
print("Upload Completed")
print(f"Documents Processed : {len(files)}")
print(f"Total Vectors : {total_vectors}")
print("=" * 60)
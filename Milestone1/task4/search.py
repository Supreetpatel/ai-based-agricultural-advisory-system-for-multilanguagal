import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

if INDEX_NAME is None:
    raise ValueError("INDEX_NAME environment variable is not set.")



print("Loading Embedding Model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Model Loaded Successfully.\n")

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

print(f"Connected to {INDEX_NAME}\n")

while True:

    query = input("\nEnter your query (or 'exit'): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    print("\n" + "=" * 80)
    print("Top Matches")
    print("=" * 80)

    if not results["matches"]:
        print("No matching documents found.")
        continue

    for i, match in enumerate(results["matches"], start=1):

        metadata = match["metadata"]

        print(f"\nResult {i}")
        print("-" * 80)
        print(f"Score      : {match['score']:.4f}")
        print(f"Source     : {metadata.get('source')}")
        print(f"Chunk No   : {metadata.get('chunk')}")
        print("\nText:\n")
        print(metadata.get("text"))
        print("-" * 80)
import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME is None:
    raise ValueError("INDEX_NAME is not set")

index = pc.Index(INDEX_NAME)

document = input("Enter document name (Example: ai.pdf): ").strip()

print("\nDeleting vectors...")

index.delete(
    filter={
        "source": document
    }
)

print("Vectors deleted.")

json_file = Path("vector_store") / f"{Path(document).stem}.json"

if json_file.exists():
    json_file.unlink()
    print("JSON removed.")

print("\nDelete Successful.")
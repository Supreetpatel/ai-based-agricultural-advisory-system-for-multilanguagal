import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

if not INDEX_NAME:
    raise ValueError("INDEX_NAME environment variable is not set.")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [
    index["name"]
    for index in pc.list_indexes()
]

if INDEX_NAME not in existing_indexes:

    print(f"Creating Index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index Created Successfully!")

else:
    print("Index already exists.")

print("Done.")
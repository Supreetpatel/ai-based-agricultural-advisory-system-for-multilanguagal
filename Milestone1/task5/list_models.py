import os
from dotenv import load_dotenv
from openai import OpenAI # pyright: ignore[reportMissingImports]

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

try:
    models = client.models.list()

    print("\nAvailable Models\n")

    model_names = sorted([model.id for model in models.data])

    for i, model in enumerate(model_names, start=1):
        print(f"{i}. {model}")

except Exception as e:
    print("Error:", e)
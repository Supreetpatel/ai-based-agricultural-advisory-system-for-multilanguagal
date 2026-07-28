import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
)

print("=" * 60)
print(" OpenAI Chatbot using LangChain")
print(" Type 'exit' to quit")
print("=" * 60)

while True:
    question = input("\nYou : ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    try:
        response = llm.invoke([HumanMessage(content=question)])

        print("\nChatGPT :", response.content)

    except Exception as e:
        print("Error :", e)
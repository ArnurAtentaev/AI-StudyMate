import os

from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import chat_vector_db
from langchain.memory import ConversationBufferWindowMemory


CONFIG = load_dotenv("../../.env")
MODEL_LLM = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")


db_client = ChromaConnect().get_connection()
collection = db_client.get_collection(name="my_collection")


llm = HuggingFaceEndpoint(
    repo_id=MODEL_LLM,
    task="conversational",
    temperature=0.5,
    top_k=10,
    top_p=0.9,
    huggingfacehub_api_token=HF_TOKEN,
)


chat_model = ChatHuggingFace(llm=llm)
model_output = chat_model.invoke("Hi my friend")


if __name__ == "__main__":
    print(model_output)

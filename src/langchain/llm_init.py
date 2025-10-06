import os
from dotenv import load_dotenv

from src.db.common_action import CommonAction

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import BM25Retriever

CONFIG = load_dotenv(".env")
MODEL_LLM = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
HF_TOKEN = os.getenv("HF_TOKEN")


model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


# llm = HuggingFaceEndpoint(
#     repo_id=MODEL_LLM,
#     task="conversational",
#     temperature=0.5,
#     top_k=10,
#     top_p=0.9,
#     huggingfacehub_api_token=HF_TOKEN,
# )
#
#
# chat_model = ChatHuggingFace(llm=llm)
# model_output = chat_model.invoke("Hi my friend")


if __name__ == "__main__":
    a = CommonAction(
        embedding_model=embedding_model,
        collection_name="documents_collection",
        persist_directory="./chroma_db",
    )
    # a.add_to_chroma(docs="./file.pdf")
    result = a.query_docs(query_text="что такое подзапрос")
    items = a.vector_db.get(include=["embeddings", "documents"])
    print(result)
    print(len(items["embeddings"]))
    print(items["embeddings"][:10])

import os
from typing import Optional, Literal
from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)


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


def initialize_llm(
    task: Optional[Literal["text-generation", "conversational"]],
    config: Optional[Literal["rag"]] = None,
):
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_LLM,
        task=task,
        temperature=0.5,
        top_k=10,
        top_p=0.9,
        huggingfacehub_api_token=HF_TOKEN,
    )
    if config == "rag":
        chat_model = ChatHuggingFace(llm=llm, verbose=True)
        return chat_model
    return llm

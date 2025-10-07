import os
from dotenv import load_dotenv

from src.db.common_action import CommonAction
from src.langgraph.states import GlobalState

from langchain_huggingface import HuggingFaceEmbeddings


CONFIG = load_dotenv(".env")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


def retriever_node(state: GlobalState, collection_name):
    db = CommonAction(embedding_model=embedding_model, collection_name=collection_name)
    db_results = db.query_docs(query_text=state["query"], chain=)
    state["results"] = db_results
    return state


def web_searcher_node(state: GlobalState):
    return

import os
from typing import Optional, Literal, List
from dotenv import load_dotenv

from src.db.common_action import CommonAction

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import BaseOutputParser
import logging


logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


class LineListOutputParser(BaseOutputParser[List[str]]):

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))


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
    config: Optional[Literal["chat"]] = None,
):
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_LLM,
        task=task,
        temperature=0.5,
        top_k=10,
        top_p=0.9,
        huggingfacehub_api_token=HF_TOKEN,
    )
    if config == "chat":
        chat_model = ChatHuggingFace(llm=llm, verbose=True)
        return chat_model
    return llm


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — ИИ-ассистент для поиска информации по документам в базе данных Chroma. "
            "Сгенерируй 3 варианта исходного вопроса, чтобы улучшить поиск документов. "
            "Варианты должны быть разными, но сохранять суть основного вопроса.",
        ),
        ("human", "Сгенерируй 3 разных переформулировки вопроса: {question}"),
    ]
)
llm = initialize_llm(task="text-generation", config="chat")
llm_chain = prompt | llm | LineListOutputParser()

action_db = CommonAction(
    embedding_model=embedding_model, collection_name="docs_collection"
)

if __name__ == "__main__":
    # action_db.add_to_chroma(docs="file.pdf")
    res = action_db.query_docs(query_text="Что такое подзапрос", chain=llm_chain)
    print(res)
    print("=================================================================")

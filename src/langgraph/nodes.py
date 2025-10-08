import os
from dotenv import load_dotenv

from src.db.common_action import CommonAction
from src.langgraph.states import GlobalState
from src.utils.parsers import LineListOutputParser
from src.langchain.llm_init import initialize_llm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import UnstructuredURLLoader


CONFIG = load_dotenv(".env")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SERPER_API = os.getenv("SERPER_API_KEY")

model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

prompt_for_retrieves = ChatPromptTemplate.from_messages(
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


def get_question_node(state: GlobalState):
    user_query = input("Введите запрос: ")
    state.query = user_query
    return state


def retriever_node(state: GlobalState, collection_name):
    db = CommonAction(embedding_model=embedding_model, collection_name=collection_name)
    llm = initialize_llm(task="text-generation", temperature=0.5, config="search")
    llm_chain = prompt_for_retrieves | llm | LineListOutputParser()

    db_results = db.query_docs(query_text=state.query, chain=llm_chain)
    state.context = db_results
    return state


def web_searcher_node(state: GlobalState):
    serper_conn = GoogleSerperAPIWrapper(
        k=5, serper_api_key=SERPER_API, result_key_for_type={"search": "organic"}
    )
    res_search = serper_conn.results(query=state.query)

    snippets = [item["snippet"] for item in res_search.get("organic", [])]
    all_info = "\n\n".join(snippets)

    state.context = all_info
    return state




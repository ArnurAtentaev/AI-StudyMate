import os
from typing import List
from dotenv import load_dotenv

from src.db.common_action import CommonAction
from src.langgraph.states import GlobalState
from src.utils.parsers import LineListOutputParser
from src.langchain.llm_init import initialize_llm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.memory import ConversationBufferWindowMemory


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
    state.question = user_query
    return state


def retriever_node(state: GlobalState, collection_name, task: str):
    db = CommonAction(embedding_model=embedding_model, collection_name=collection_name)
    llm = initialize_llm(task=task, temperature=0.5, config="search")
    llm_chain = prompt_for_retrieves | llm | LineListOutputParser()

    db_results: List[Document] = db.query_docs(
        query_text=state.question, chain=llm_chain
    )
    combined_text = "\n\n".join([doc.page_content for doc in db_results])

    state.context = combined_text
    return state


def web_searcher_node(state: GlobalState):
    serper_conn = GoogleSerperAPIWrapper(
        k=5, serper_api_key=SERPER_API, result_key_for_type={"search": "organic"}
    )
    res_search = serper_conn.results(query=state.question)

    snippets = [item["snippet"] for item in res_search.get("organic", [])]
    all_info = "\n\n".join(snippets)

    state.context = all_info
    return state


def conversational_node(state: GlobalState):
    """Формирует человекоподобный финальный ответ на основе результата RAG и истории."""
    conv_prompt = ChatPromptTemplate.from_template(
        """
    Ты полезный ассистент. Который использует данные с RAG(Retriever Augmented Generation) и может подстраиваться под язык, на котором к тебе обращаются тем самым меняя язык на котором ты мыслишь и действуешь.
    Важно: если в тексте встречаются технические термины (например: dict, class, agent, prompt, pipeline и т. д.), то ты НЕ переводишь их на русский, а оставляешь в оригинале.
    Также ты прикладываешь примеры, если они есть, не придумывай сам, а используй примеры которые ты находишь при поиске..
    На основе RAG результата и контекста сформулируй понятный и точный ответ.
    RAG result:
    {rag_result}

    Вопрос:
    {question}
    """
    )
    llm_conv = initialize_llm(task="conversational", temperature=1, config="search")

    chain = conv_prompt | llm_conv
    final_answer = chain.invoke(
        {"rag_result": state.context, "question": state.question}
    )
    state.answer = final_answer
    return state


state = GlobalState(question="Что делает FROM в SQL?")
context = retriever_node(
    state=state, collection_name="docs_collection", task="text-generation"
)
res = conversational_node(context)

print(res.answer)

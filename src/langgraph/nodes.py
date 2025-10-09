import os
from typing import List
from dotenv import load_dotenv

from src.db.common_action import CommonAction
from src.langgraph.states import GlobalState, GradeDocuments
from src.utils.parsers import LineListOutputParser
from src.langchain.llm_init import initialize_llm
from prompts import PROMPT_FOR_RETRIEVER, PROMPT_FOR_ANSWER, PROMPT_FOR_GRADE_RESULT_RAG

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
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
        ("system", PROMPT_FOR_RETRIEVER),
        ("human", "Сгенерируй 3 разных переформулировки вопроса: {question}"),
    ]
)


def get_question_node(state: GlobalState, text) -> GlobalState:
    state.question = text
    return state


def retriever_node(state: GlobalState, collection_name, task: str) -> GlobalState:
    db = CommonAction(embedding_model=embedding_model, collection_name=collection_name)
    llm = initialize_llm(task=task, temperature=0.5, config="search")
    llm_chain = prompt_for_retrieves | llm | LineListOutputParser()

    db_results: List[Document] = db.query_docs(
        query_text=state.question, chain=llm_chain
    )
    combined_text = "\n\n".join([doc.page_content for doc in db_results])

    state.result_rag = combined_text
    return state


def grade_rag_result(state):
    documents = state.result_rag

    llm = initialize_llm(task="text-generation", temperature=0.5, config="search")
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_FOR_GRADE_RESULT_RAG),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader
    filtered_documents = []

    for document in documents:
        grader_response = retrieval_grader.invoke(
            {"question": state.question, "document": documents}
        )
        if grader_response.binary_score.lower() == "yes":
            filtered_documents.append(document)

    state.relevant_documents = filtered_documents
    return state


def decide_to_generate(state):
    if len(state.relevant_documents) > 0:
        return "generate"
    else:
        return "transform_query"


def web_searcher_node(state: GlobalState) -> GlobalState:
    serper_conn = GoogleSerperAPIWrapper(
        k=5, serper_api_key=SERPER_API, result_key_for_type={"search": "organic"}
    )
    res_search = serper_conn.results(query=state.question)

    snippets = [item["snippet"] for item in res_search.get("organic", [])]
    all_info = "\n\n".join(snippets)

    state.result_google = all_info
    return state


def conversational_node(state: GlobalState):
    """Формирует человекоподобный финальный ответ на основе результата RAG или если нет подходящих данных в базе данных, тебе подается результат поиска в Google и истории."""
    conv_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_FOR_ANSWER),
            ("human", "RAG result:\n{rag_result}\n" "Вопрос:\n{question}"),
        ]
    )
    llm_conv = initialize_llm(task="conversational", temperature=1, config="search")

    chain = conv_prompt | llm_conv
    final_answer = chain.invoke(
        {"rag_result": state.result_rag, "question": state.question}
    )
    state.answer = final_answer
    return state


state = GlobalState()
query = get_question_node(state=state, text="как получить данные из подзапроса?")
context = retriever_node(
    state=state, collection_name="docs_collection", task="text-generation"
)
res = conversational_node(context)

print(res.answer)

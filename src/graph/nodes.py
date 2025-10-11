import os
from typing import List
from dotenv import load_dotenv

from src.db.common_action import CommonAction
from src.langgraph.states import GlobalState, GradeDocuments
from src.utils.parsers import LineListOutputParser
from src.langchain.llm_init import initialize_llm
from src.langgraph.prompts import (
    PROMPT_FOR_RETRIEVER,
    PROMPT_FOR_ANSWER,
    PROMPT_FOR_GRADE_RESULT_RAG,
    QUESTION_REWRITER_PROMPT,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
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


def retriever_node(state: GlobalState) -> GlobalState:
    """
    Get relevant documents from the vector store.
    """
    db = CommonAction(
        embedding_model=embedding_model, collection_name="docs_collection"
    )

    prompt_for_retrieves = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_FOR_RETRIEVER),
            ("human", "Сгенерируй 3 разных переформулировки вопроса: {question}"),
        ]
    )

    llm = initialize_llm(temperature=0.5)
    llm_chain = prompt_for_retrieves | llm | LineListOutputParser()

    db_results: List[Document] = db.query_docs(
        query_text=state.question, chain=llm_chain
    )
    combined_text = [doc.page_content for doc in db_results]

    state.result_rag = combined_text
    print("retriever done")
    return state


def grade_rag_result(state) -> GlobalState:
    documents = state.result_rag

    llm = initialize_llm(temperature=0.5)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_FOR_GRADE_RESULT_RAG),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    retrieval_grader = grade_prompt | llm
    filtered_documents = []

    for document in documents:
        grader_response = retrieval_grader.invoke(
            {"question": state.question, "document": document}
        )

        response_text = grader_response.content.strip().lower()

        if "yes" in response_text or "да" in response_text:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_documents.append(document)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    print("Relevant documents left after filtering:", len(filtered_documents))
    state.relevant_documents = filtered_documents
    print("relevant check retriever done")
    return state


def generate_answer_from_documents_node(state: GlobalState) -> GlobalState:
    """
    Forms a human-like final answer based on the RAG result or if there is no suitable data in the database,
    you are given the search result in Google and history.
    """
    conv_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_FOR_ANSWER),
            ("human", "RAG result:\n{result}\n" "Вопрос:\n{question}"),
        ]
    )
    llm_conv = initialize_llm(temperature=1)

    chain = conv_prompt | llm_conv

    if len(state.result_rag) == 0:
        final_answer = chain.invoke(
            {"result": state.result_google, "question": state.question}
        )
        state.answer = final_answer
        return state

    final_answer = chain.invoke(
        {"result": state.result_rag, "question": state.question}
    )
    state.answer = final_answer
    return state


def decide_to_generate(state):
    """Decide whether to generate an answer or perform a web search."""
    if len(state.relevant_documents) > 0:
        print("\n Найдены релевантные документы, генерирую ответ\n\n")
        return "generate"
    else:
        print(
            "\n Нет релевантных документов в базе данных, начинаю выполнять поиск в интернете... \n\n"
        )
        return "transform_query"


def transform_query(state) -> GlobalState:
    """
    Transform the query to produce a better question.
    """

    print("\n\n ---TRANSFORMING QUERY---")
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUESTION_REWRITER_PROMPT),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    llm = initialize_llm(temperature=0.7)
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    better_question = question_rewriter.invoke({"question": state.question})

    state.question = better_question
    return state


def web_searcher_node(state: GlobalState) -> GlobalState:
    """This is a web search"""

    print("-----STARTED WEB SEARCH-----")
    serper_conn = GoogleSerperAPIWrapper(
        k=5, serper_api_key=SERPER_API, result_key_for_type={"search": "organic"}
    )
    res_search = serper_conn.results(query=state.question)

    snippets = [item["snippet"] for item in res_search.get("organic", [])]
    all_info = "\n\n".join(snippets)

    state.result_google = all_info
    return state

# state = GlobalState(question="Что такое библия?")
# 
# transf_question = transform_query(state)
# print(transf_question)
# search = web_searcher_node(transf_question)
# print(search)


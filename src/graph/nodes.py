import os
import logging
from typing import List
from dotenv import load_dotenv

from src.db.common_action import CommonAction
from src.graph.chat_memory import db_connection
from src.graph.states import GlobalState
from src.utils.parsers import LineListOutputParser
from src.langchain_utils.llm_init import initialize_llm
from src.graph.prompts import (
    PROMPT_FOR_RETRIEVER,
    PROMPT_FOR_ANSWER,
    PROMPT_FOR_GRADE_RESULT_RAG,
    QUESTION_REWRITER_PROMPT,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser


log = logging.getLogger("nodes_logging")
log.setLevel(logging.INFO)

CONFIG = load_dotenv(".env")
SERPER_API = os.getenv("SERPER_API_KEY")


def retriever_node(state: GlobalState) -> GlobalState:
    """
    Get relevant documents from the vector store.
    """
    db = CommonAction(collection_name="docs_collection")

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
    logging.info("----RETRIEVER FINISHED-----")
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
            logging.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_documents.append(document)
        else:
            logging.info("---GRADE: DOCUMENT NOT RELEVANT---")

    logging.info(f"Relevant documents left after filtering: {len(filtered_documents)}")
    state.relevant_documents = filtered_documents
    logging.info("-----RETRIEVER RESPONSE RELEVANCE CHECK COMPLETED-----")
    return state


def generate_answer_node(state: GlobalState) -> GlobalState:
    """
    Forms a human-like final answer based on the RAG result or if there is no suitable data in the database,
    you are given the search result in Google and history.
    """
    logging.info("-----ANSWER GENERATION STARTED-----")
    conv_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_FOR_ANSWER),
            (
                "human",
                "RAG result:\n{result}\n"
                "Вопрос:\n{question}"
                "История чата:\n{history}\n",
            ),
        ]
    )
    llm_conv = initialize_llm(temperature=1)

    db_history = db_connection(session_id=state.session_id)
    state.history = db_history.get_messages()
    formatted_history = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in state.history]
    )
    config = {"configurable": {"session_id": state.session_id}}

    chain = conv_prompt | llm_conv

    result_source = (
        state.result_rag if len(state.result_rag) > 0 else state.result_google
    )
    final_answer = chain.invoke(
        {
            "result": result_source,
            "question": state.question,
            "history": formatted_history,
        },
        config=config,
    )
    state.answer = final_answer
    db_history.add_messages(
        [
            HumanMessage(content=state.question),
            AIMessage(content=final_answer.content),
        ]
    )

    logging.info("-----ANSWER GENERATED -----")
    print(len(db_history.messages))
    return state


def decide_to_generate(state):
    """Decide whether to generate an answer or perform a web search."""
    logging.info("-----THINKING ABOUT WHICH TOOL TO CHOOSE-----")
    if len(state.relevant_documents) > 0:
        logging.info("\n Relevant documents found, generating answer\n\n")
        return "generate"
    else:
        logging.info(
            "\n No relevant documents in the database, start searching on the internet... \n\n"
        )
        return "transform_query"


def transform_query(state) -> GlobalState:
    """
    Transform the query to produce a better question.
    """

    logging.info("\n\n -----TRANSFORMING QUERY-----")
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

    logging.info("-----STARTED WEB SEARCH-----")
    serper_conn = GoogleSerperAPIWrapper(
        k=5, serper_api_key=SERPER_API, result_key_for_type={"search": "organic"}
    )
    res_search = serper_conn.results(query=state.question)

    snippets = [item["snippet"] for item in res_search.get("organic", [])]
    all_info = "\n\n".join(snippets)

    state.result_google = all_info
    return state

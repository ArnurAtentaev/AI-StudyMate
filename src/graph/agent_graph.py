from src.graph.states import GlobalState
from src.graph.nodes import (
    retriever_node,
    grade_rag_result,
    generate_answer_node,
    decide_to_generate,
    web_searcher_node,
    transform_query,
)

from langgraph.graph import StateGraph, START, END


def build_graph():
    workflow = StateGraph(GlobalState)

    workflow.add_node("get_retriever_result", retriever_node)
    workflow.add_node("check_retriever_result", grade_rag_result)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("web_search", web_searcher_node)
    workflow.add_node("transform_query", transform_query)

    workflow.add_edge(START, "get_retriever_result")
    workflow.add_edge("get_retriever_result", "check_retriever_result")
    workflow.add_conditional_edges(
        "check_retriever_result",
        decide_to_generate,
        {
            "generate": "generate_answer",
            "transform_query": "transform_query",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()

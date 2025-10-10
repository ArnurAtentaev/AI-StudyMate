from src.langgraph.states import GlobalState
from src.langgraph.nodes import (
    retriever_node,
    grade_rag_result,
    generate_answer_from_documents_node,
    decide_to_generate,
    web_searcher_node,
    transform_query,
)

from langgraph.graph import StateGraph, START, END


def build_graph():
    workflow = StateGraph(GlobalState)

    workflow.add_node("get_retriever_result", retriever_node)
    workflow.add_node("check_retriever_result", grade_rag_result)
    workflow.add_node("rag_based_answer", generate_answer_from_documents_node)
    workflow.add_node("web_search", web_searcher_node)
    workflow.add_node("transform_query", transform_query)

    workflow.add_edge(START, "get_retriever_result")
    workflow.add_edge("get_retriever_result", "check_retriever_result")
    workflow.add_conditional_edges(
        "check_retriever_result",
        decide_to_generate,
        {
            "generate": "rag_based_answer",
            "transform_query": "transform_query",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "rag_based_answer")
    workflow.add_edge("rag_based_answer", END)

    return workflow.compile()

compiled_graph = build_graph()
print("***********************************************************************************************")
# print(compiled_graph.get_graph().draw_mermaid())
state = compiled_graph.invoke({"question": "Что такое библия?"})
print("\nОтвет модели: \n")
print(state["answer"])

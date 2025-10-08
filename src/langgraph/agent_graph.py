from src.langgraph.states import GlobalState
from src.langgraph.nodes import get_question_node, retriever_node, web_searcher_node

from langgraph.graph import StateGraph, START, END


graph_workflow = StateGraph(GlobalState)
graph_workflow.add_node()
graph_workflow.add_node()

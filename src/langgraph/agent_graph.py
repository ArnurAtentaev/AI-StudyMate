from src.langgraph.states.web_states import StatesWebSearcher

from langgraph.graph import StateGraph, END


graph_workflow = StateGraph(StatesWebSearcher)
graph_workflow.add_node(node="first node")


from src.langgraph.states import GlobalState

from langgraph.graph import StateGraph, START, END


graph_workflow = StateGraph(GlobalState)
graph_workflow.add_node()
graph_workflow.add_node()

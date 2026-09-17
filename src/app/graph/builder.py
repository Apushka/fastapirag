from langgraph.graph import StateGraph, START
from src.app.graph.state import State
from src.app.graph.nodes import call_model
from src.app.graph.tools import tool_node

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", lambda state: "tools" if state.messages[-1].tool_calls else "__end__"
)
builder.add_edge("tools", "agent")

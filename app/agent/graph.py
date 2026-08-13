from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    classify_node,
    general_node,
    medication_node,
    mixed_node,
    pharmacy_node,
    safety_node,
)
from app.agent.state import AgentState
from app.memory.checkpointer import memory


def route_by_intent(state: AgentState) -> str:
    return state["intent"]


def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("classify", classify_node)
    graph_builder.add_node("pharmacy", pharmacy_node)
    graph_builder.add_node("medication", medication_node)
    graph_builder.add_node("mixed", mixed_node)
    graph_builder.add_node("safety", safety_node)
    graph_builder.add_node("general", general_node)

    graph_builder.add_edge(START, "classify")

    graph_builder.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "pharmacy": "pharmacy",
            "medication": "medication",
            "mixed": "mixed",
            "safety": "safety",
            "general": "general",
        },
    )

    graph_builder.add_edge("pharmacy", END)
    graph_builder.add_edge("medication", END)
    graph_builder.add_edge("mixed", END)
    graph_builder.add_edge("safety", END)
    graph_builder.add_edge("general", END)

    return graph_builder.compile(checkpointer=memory)


graph = build_graph()

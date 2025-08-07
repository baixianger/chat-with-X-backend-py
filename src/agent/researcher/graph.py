"""
Researcher Graph.
"""
# pylint: disable=wrong-import-position
import os
import sys
from langgraph.graph import END, START, StateGraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.agent.researcher.tools.queries import generate_queries
from src.agent.researcher.tools.retriever import (
    retrieve_documents,
    retrieve_in_parallel,
)
from src.agent.researcher.state import ResearcherState


# Define the graph
builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_edge(START, "generate_queries")
builder.add_node(retrieve_documents) # type: ignore
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_documents"],
)
builder.add_edge("retrieve_documents", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "ResearcherGraph"

if __name__ == "__main__":   
    input_state = ResearcherState(
        step="How to build a chatbot?",
        collections=["langchain"]
    )
    graph.invoke(input_state)

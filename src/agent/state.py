"""
State for the retrieval graph.
"""
# pylint: disable=wrong-import-position
import os
import sys
from typing import Literal, Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langchain_core.documents import Document
from langgraph.graph import add_messages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import reduce_docs


class RouterState(BaseModel):
    """State for the router and the structured output of the LLM in the router."""
    logic: str = Field(
        default="", description="The logic/justification of the router result."
    )
    type: Literal["more-info", "related", "unrelated", "chitchat"] = Field(
        default="chitchat",
        description="The type of the router."
    )
    collections: Optional[list[str]] = Field(
        default=None,
        description="The list of collections related to the query, at most 2 collections.",
    )
    response: Optional[str] = Field(
        default=None,
        description="The response to the user, if the query is a unrelated, more-info or chitchat.",
    )


class Plan(BaseModel):
    """Steps in a research plan."""
    steps: list[str] = Field(
        default_factory=list,
        description="A list of steps in the research plan, at most 3 steps.",
    )


class InputState(BaseModel):
    """Input state for the retrieval graph."""
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list, description="Accumulated messages with unique IDs."
    )


class AgentState(InputState, RouterState):
    """State for the agent."""
    question: str = Field(default="", description="The latest question from the user.")
    steps: list[str] = Field(
        default_factory=list, description="A list of steps in the research plan."
    )
    documents: Annotated[list[Document], reduce_docs] = Field(
        default_factory=list,
        description="Populated by the retriever. " +
                    "This is a list of documents that the agent can reference.",
    )
    answer: str = Field(default="", description="Final answer. Useful for evaluations")

"""
Configuration for the retrieval graph.
"""
# pylint: disable=wrong-import-position
import os
import sys
from typing import Literal, Any, Annotated
from pydantic import BaseModel, Field

# Add the project root to sys.path for relative imports
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")))
from src import prompts

class LLMConfig(BaseModel):
    """Configuration for the LLM."""
    query_model: str = Field(
        default="openai/gpt-4o-mini",
        description=(
            "The language model used for processing and refining queries. "
            "Should be in the form: provider/model-name."
        ),
    )
    response_model: str = Field(
        default="openai/gpt-4o-mini",
        description=(
            "The language model used for generating responses. "
            "Should be in the form: provider/model-name."
        ),
    )


class EmbeddingsConfig(BaseModel):
    """Configuration for the embeddings."""
    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = Field(
        default="openai/text-embedding-3-small",
        description="Name of the embedding model to use. Must be a valid embedding model name.",
    )


class RetrieverConfig(EmbeddingsConfig):
    """Configuration for the retriever."""
    retriever_provider: Annotated[
        Literal["chroma", "weaviate", "duck", "supabase"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = Field(
        default="chroma",
        description="Name of the retriever to use. Must be a valid retriever name.",
    )

    storage_type: Literal["persistent", "ephemeral", "cloud", "local"] = Field(
        default="persistent",
        description="The type of storage to use for the retriever.",
    )

    chunk_size: int = Field(
        default=4000,
        description="The maximum number of tokens in a chunk.",
    )

    chunk_overlap: int = Field(
        default=200,
        description="The number of tokens to overlap between chunks.",
    )

    search_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the search function of the retriever.",
    )


class PromptConfig(BaseModel):
    """Configuration for the prompts."""

    router_system_prompt: str = Field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        description=(
            "The system prompt used for classifying user questions "
            "to route them to the correct node."
        ),
    )

    research_plan_system_prompt: str = Field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        description=(
            "The system prompt used for generating a research plan based on the user's question."
        ),
    )

    generate_queries_system_prompt: str = Field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        description=(
            "The system prompt used by the researcher to generate queries "
            "based on a step in the research plan."
        ),
    )

    response_system_prompt: str = Field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        description="The system prompt used for generating responses.",
    )


class Configuration(LLMConfig, RetrieverConfig, PromptConfig):
    """Configuration for the retrieval graph."""


if __name__ == "__main__":
    config = Configuration()
    print(config)

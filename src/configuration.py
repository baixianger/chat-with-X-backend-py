import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydantic import BaseModel, Field
from typing import Literal, Any, Annotated
from src import prompts


class LLMConfig(BaseModel):
    query_model: str = Field(
        default="openai/gpt-4o-mini",
        description="The language model used for processing and refining queries. Should be in the form: provider/model-name.",
    )
    response_model: str = Field(
        default="openai/gpt-4o-mini",
        description="The language model used for generating responses. Should be in the form: provider/model-name.",
    )


class EmbeddingsConfig(BaseModel):
    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = Field(
        default="openai/text-embedding-3-small",
        description="Name of the embedding model to use. Must be a valid embedding model name.",
    )


class RetrieverConfig(EmbeddingsConfig):
    retriever_provider: Annotated[
        Literal["chroma", "weaviate", "duck", "supabase"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = Field(
        default="chroma",
        description="Name of the retriever to use. Must be a valid retriever name.",
    )

    search_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )


class PromptConfig(BaseModel):

    router_system_prompt: str = Field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        description="The system prompt used for classifying user questions to route them to the correct node.",
    )

    more_info_system_prompt: str = Field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        description="The system prompt used for asking for more information from the user.",
    )

    general_system_prompt: str = Field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        description="The system prompt used for responding to general questions.",
    )

    research_plan_system_prompt: str = Field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        description="The system prompt used for generating a research plan based on the user's question.",
    )

    generate_queries_system_prompt: str = Field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        description="The system prompt used by the researcher to generate queries based on a step in the research plan.",
    )

    response_system_prompt: str = Field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        description="The system prompt used for generating responses.",
    )


class Configuration(LLMConfig, RetrieverConfig, PromptConfig):
    pass


if __name__ == "__main__":
    config = Configuration()
    print(config)

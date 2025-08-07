"""
Query tools for the researcher.
"""
# pylint: disable=wrong-import-position
import os
import sys
from typing import cast
from langchain_core.runnables import RunnableConfig

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)

from src.utils import load_chat_model
from src.configuration import Configuration
from src.agent.researcher.state import GeneratedQueries, ResearcherState


async def generate_queries(
    state: ResearcherState, config: RunnableConfig
) -> GeneratedQueries:
    """Generate search queries based on the research step (a step in the research plan)."""

    configuration = Configuration.model_validate(
        config.get("configurable")
    )
    structured_output_kwargs = (
        {"method": "function_calling"} if "openai" in configuration.query_model else {}
    )
    model, _, _ = load_chat_model(configuration.query_model)
    model = model.with_structured_output(GeneratedQueries, include_raw=False, **structured_output_kwargs)

    system_prompt = configuration.generate_queries_system_prompt.format(
        collections=state.collections
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": state.step},
    ]

    generated_queries = cast(
        GeneratedQueries,
        await model.ainvoke(messages, {"tags": ["langsmith:nostream"]}),
    )
    return generated_queries

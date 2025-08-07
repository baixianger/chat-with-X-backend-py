"""
Retriever tools for the researcher.
"""
# pylint: disable=wrong-import-position
import os
import sys
from typing import Iterator, cast
from contextlib import contextmanager
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)
from src.embeddings import get_embeddings_model
from src.vectorstore import get_vector_store
from src.configuration import RetrieverConfig
from src.agent.researcher.state import QueryState, ResearcherState


@contextmanager
def make_retriever(
    collection_name: str,
    config: RunnableConfig,
) -> Iterator[BaseRetriever]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = RetrieverConfig.model_validate(config.get("configurable"))
    store = get_vector_store(
        configuration.retriever_provider,
        configuration.storage_type,
        collection_name=collection_name,
        embedding=get_embeddings_model(configuration.embedding_model),
    )
    search_kwargs = configuration.search_kwargs
    yield store.as_retriever(search_kwargs=search_kwargs)


async def retrieve_documents(
    state: QueryState, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query."""
    documents = []
    for collection_name in state.collections:
        with make_retriever(collection_name, config) as retriever:
            response = await retriever.ainvoke(state.query, config)
            documents.extend(response)
    return {"documents": documents}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query."""
    return [
        Send(
            "retrieve_documents",
            cast(QueryState, query),
        )
        for query in state.queries
    ]

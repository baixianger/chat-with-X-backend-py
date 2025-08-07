"""
Vector store for the retrieval graph.
"""
# pylint: disable=wrong-import-position
import os
import sys
from typing import Literal
import chromadb
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.utils import get_vector_db_dir


def get_collection_list(provider: Literal["chroma", "duck", "weaviate", "supabase"]):
    """Get collection list from vector db client"""
    if provider == "chroma":
        client: ClientAPI = chromadb.PersistentClient(path=get_vector_db_dir(provider))
        return [collection.name for collection in client.list_collections()]
    else:
        raise ValueError(f"We will add support for {provider} in the future")


def get_vector_store(
    provider: Literal["chroma", "duck", "weaviate", "supabase"],
    storage_type: Literal["persistent", "ephemeral", "cloud", "local"],
    collection_name: str,
    embedding: Embeddings,
    **kwargs,
) -> Chroma:
    """Get vector store from vector db client"""
    path = get_vector_db_dir(provider)
    if provider == "chroma":
        if storage_type == "persistent":
            client: ClientAPI = chromadb.PersistentClient(path=path)
        else:
            raise ValueError(
                f"We will add support for running chromadb in {storage_type} mode in the future"
            )
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding,
            **kwargs,
        )
    else:
        raise ValueError(f"We will add support for {provider} in the future")

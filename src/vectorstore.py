import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import chromadb
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from typing import Literal
from src.utils import get_vector_db_dir


def get_vector_store(
    provider: Literal["chroma"],
    model: Literal["persistent", "ephemeral", "cloud", "local"],
    collection_name: str,
    embedding: Embeddings,
    **kwargs,
) -> Chroma:
    path = get_vector_db_dir(provider)
    if provider == "chroma":
        if model == "persistent":
            client: ClientAPI = chromadb.PersistentClient(path=path)
        else:
            raise ValueError(
                f"We will add support for running chromadb in {model} mode in the future"
            )
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding,
            **kwargs,
        )
    else:
        raise ValueError(f"We will add support for {provider} in the future")

"""
Embeddings for the retrieval graph.
"""
# pylint: disable=wrong-import-position
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings_model(model: str = "openai/text-embedding-3-small") -> Embeddings:
    """Get the embeddings model."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            return OpenAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")

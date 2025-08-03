from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings_model(model: str = "text-embedding-3-small") -> Embeddings:
    return OpenAIEmbeddings(model=model)

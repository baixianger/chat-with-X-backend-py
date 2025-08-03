import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import chromadb
import weaviate
import duckdb
import supabase
from langchain_community.vectorstores import ChromaVectorStore
from langchain_community.vectorstores import WeaviateVectorStore
from langchain_community.vectorstores import DuckDBVectorStore
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.embeddings import Embeddings
from typing import Literal

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../",
    os.environ.get("DATA_PATH", "./data"),
)


def get_vector_store(
    provider: Literal["chroma", "weaviate", "duck", "supabase"],
    collection_name: str,
    embedding: Embeddings,
) -> BaseVectorStore:
    if provider == "chroma":
        client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "/chroma_db"))
        store = ChromaVectorStore(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding,
        )
    elif provider == "weaviate":
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=weaviate.classes.init.Auth.api_key(
                os.environ.get("WEAVIATE_API_KEY", "not_provided")
            ),
            skip_init_checks=True,
        )
        store = WeaviateVectorStore(
            client=client,
            index_name=collection_name,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )
    elif provider == "duck":
        client = duckdb.connect(os.path.join(DATA_DIR, "duck_db/duckDB"))
        store = DuckDBVectorStore(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding,
            attributes=["source", "title"],
        )
    elif provider == "supabase":
        client = supabase.connect(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"]
        )
        store = SupabaseVectorStore(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding,
            attributes=["source", "title"],
        )

    return store

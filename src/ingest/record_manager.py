"""
Record manager for the retrieval graph.
"""
# pylint: disable=wrong-import-position
import os
import sys
from typing import Literal
from langchain.indexes import SQLRecordManager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import get_record_db_url



def get_record_manager(
    vector_provider: Literal["chroma", "supabase", "weaviate", "duck"],
    collection_name: str,
    embedding_name: str,
):
    """Get the record manager."""
    namespace = f"{vector_provider}/{collection_name}/{embedding_name}"
    return SQLRecordManager(
        namespace=namespace,
        db_url=get_record_db_url(),
    )


if __name__ == "__main__":
    print(get_record_db_url())
    print(get_record_manager("chroma", "test_collection", "text-embedding-3-small"))

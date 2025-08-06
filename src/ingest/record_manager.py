import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from langchain.indexes import SQLRecordManager
from src.utils import get_record_db_url
from typing import Literal


def get_record_manager(
    vector_provider: Literal["chroma", "supabase", "weaviate", "duck"],
    collection_name: str,
    embedding_name: str,
):
    namespace = f"{vector_provider}/{collection_name}/{embedding_name}"
    return SQLRecordManager(
        namespace=namespace,
        db_url=get_record_db_url(),
    )


if __name__ == "__main__":
    print(get_record_db_dir())
    print(get_record_manager("chroma", "test_collection", "text-embedding-3-small"))

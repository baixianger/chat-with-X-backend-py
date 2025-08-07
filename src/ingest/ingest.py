"""
Ingest data into the retrieval graph.
"""
# pylint: disable=wrong-import-position
import os
import sys
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import index

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.embeddings import get_embeddings_model
from src.configuration import Configuration
from src.vectorstore import get_vector_store
from src.ingest.record_manager import get_record_manager
from src.ingest.doc_loader import recursive_url_loader
from src.ingest.parsers.langchain_recursive_url import (
    langchain_recursive_url_metadata_extractor,
    langchain_recursive_url_extractor,
    exclude_urls_doc,
    exclude_urls_ref,
)



def ingest(collection_name: str):
    """Ingest data into the retrieval graph."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = Configuration()
    embedding = get_embeddings_model(config.embedding_model)
    store = get_vector_store(
        provider=config.retriever_provider,
        storage_type=config.storage_type,
        collection_name=collection_name,
        embedding=embedding,
    )
    record_manager = get_record_manager(
        vector_provider=config.retriever_provider,
        collection_name=collection_name,
        embedding_name=config.embedding_model,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )

    langchain_doc = recursive_url_loader(
        path="https://python.langchain.com/docs/",
        filter_urls=exclude_urls_doc,
        max_depth=4,
        meta_kwargs={"doc_type": "doc", "lang": "python"},
        metadata_extractor=langchain_recursive_url_metadata_extractor,
        extractor=langchain_recursive_url_extractor,
    )
    logger.info("Loaded %d docs from documentation", len(langchain_doc))

    langchain_ref = recursive_url_loader(
        path="https://python.langchain.com/api_reference/",
        filter_urls=exclude_urls_ref,
        max_depth=4,
        meta_kwargs={"doc_type": "ref", "lang": "python"},
        metadata_extractor=langchain_recursive_url_metadata_extractor,
        extractor=langchain_recursive_url_extractor,
    )
    logger.info("Loaded %d docs from api_reference", len(langchain_ref))

    langchain_code = recursive_url_loader(
        path="https://python.langchain.com/api_reference/_modules/",
        max_depth=4,
        meta_kwargs={"doc_type": "code", "lang": "python"},
        metadata_extractor=langchain_recursive_url_metadata_extractor,
        extractor=langchain_recursive_url_extractor,
    )
    logger.info("Loaded %d docs from source_code", len(langchain_code))

    docs_transformed = text_splitter.split_documents(
        langchain_doc + langchain_ref + langchain_code
    )

    indexing_stats = index(
        docs_transformed,
        record_manager,
        store,
        cleanup="full",  # 旧的有的，新的没有的，会删除
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info("Indexing stats: %s", indexing_stats)
    # Get collection count using public API
    num_vecs = store._collection.count()  # pylint: disable=protected-access
    logger.info(
        "LangChain now has this many vectors: %d",
        num_vecs,
    )


if __name__ == "__main__":

    # load .env
    from dotenv import load_dotenv

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    load_dotenv(dotenv_path=os.path.join(base_dir, ".env"), override=True)
    ingest("langchain")

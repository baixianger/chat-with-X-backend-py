import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embeddings import get_embeddings_model
from src.configuration import Configuration
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_docs():
    CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./vectorDB/chroma_db")
    RECORD_MANAGER_DB_URL = os.environ.get("RECORD_MANAGER_DB_URL", "sqlite://")

    config = Configuration()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

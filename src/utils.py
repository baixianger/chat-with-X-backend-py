"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""
import os
import uuid
from typing import Any, Literal, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatTongyi

def get_record_db_url():
    """Get the URL of the record database."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(base_dir, "../data/recordDB")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "record_manager.db")
    return f"sqlite:///{db_path}"


def get_vector_db_dir(provider: Literal["chroma", "supabase", "weaviate", "duck"]) -> str:
    """Get the directory of the vector database."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../data/vectorDB/{provider}_db",
    )


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Reduce and process documents based on the input type."""
    if new == "delete":
        return []

    existing_list: list[Document] = list[Document](existing) if existing else []
    new_list: list[Document] = []
    if isinstance(new, str):
        new_list = [Document(page_content=new, id=str(uuid.uuid4()))]
    elif isinstance(new, list):
        existing_ids = set(doc.id for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = str(uuid.uuid4())
                new_list.append(Document(page_content=item, id=item_id))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                item_id = item.get("id", str(uuid.uuid4()))

                if item_id not in existing_ids:
                    new_list.append(Document(**item, id=item_id))
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.id
                if item_id is None:
                    item_id = str(uuid.uuid4())
                    new_item = item.model_copy(deep=True)
                    new_item.id = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list


def load_chat_model(fully_specified_name: str) -> tuple[BaseChatModel, str, str]:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, name = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        name = fully_specified_name

    if provider == "tongyi":
        # init_chat_model doesn't support tongyi
        return ChatTongyi(name=name, api_key=None, model_kwargs={"temperature": 0}), provider, name

    model_kwargs = {}
    if provider == "google_genai":
        # google doesn't support system message
        model_kwargs["convert_system_message_to_human"] = True
    return (
        init_chat_model(name, model_provider=provider, temperature=0, **model_kwargs),
        provider,
        name,
    )


if __name__ == "__main__":
    print(get_record_db_url())
    print(get_vector_db_dir("chroma"))

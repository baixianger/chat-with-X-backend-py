import os
import sys
import re
import requests
import aiohttp
from types import NoneType
from typing import Generator, Optional, Union, Literal, Callable
from bs4 import BeautifulSoup, Doctype, NavigableString, Tag

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def langchain_recursive_url_metadata_extractor(
    raw_html: str,
    url: str,
    response: Union[requests.Response, aiohttp.ClientResponse],
    *,
    type: Literal["documents", "api_reference", "source_code"],
    lang: Optional[str] = "python",
) -> dict:
    soup = BeautifulSoup(raw_html, "lxml")
    title_element = soup.find("h1")
    try:
        title_element.find("a").decompose()
    except AttributeError:
        pass
    title = (
        title_element.get_text() if title_element else url.rstrip("/").split("/")[-1]
    )
    return {
        "source": url,
        "title": title,
        "type": type,
        "lang": lang if lang else "",
    }


def get_title(title: Tag) -> Generator[str, None, None]:
    a_tag = title.find("a")
    a_tag.decompose() if a_tag else None
    yield f"{'#' * int(title.name[1:])} {title.get_text(strip=True)}\n\n"


def get_language(div: Tag) -> str:
    # highlight-<language> class is for api reference pages
    # language-<language> class is for documents pages
    classes = div.get("class", [])
    for cls in classes:
        if re.match(r"(highlight|language)-\w+", cls):
            return cls.split("-")[1]
    return ""


def get_code(pre: Tag) -> str:
    for a_tag in pre.find_all("a"):
        a_tag.decompose()
    if pre.find("code"):  # for code and doc
        code = pre.find("code")
        for child in code.children:
            yield child.get_text() + "\n"
    # for api reference, in ord to compatible to the dl dt dd structure
    else:
        lines = pre.get_text().split("\n")
        for line in lines:
            yield line + "\n"


def get_list(
    list: Tag,
    ordered: bool,
    nested_handler: Callable[[Tag], Generator[str, None, None]],
) -> Generator[str, None, None]:
    indent_str = "  "

    for i, li in enumerate(list.find_all("li", recursive=False)):
        prefix = f"{i + 1}. " if ordered else "- "
        yield f"{indent_str}{prefix}"

        # 缩进子内容
        for i, line in enumerate(nested_handler(li)):
            if isinstance(line, Tag) and line.name in ["em", "strong", "b", "i", "a"]:
                yield line
            elif isinstance(line, str):
                yield line
            else:
                yield f"{indent_str}  {line}"

        yield "\n\n"


def get_description(
    dl: Tag, nested_handler: Callable[[Tag], Generator[str, None, None]]
) -> Generator[str, None, None]:
    dt_tags = dl.find_all("dt", recursive=False)
    dd_tags = dl.find_all("dd", recursive=False)

    # Case 1: 数量匹配，一对一处理
    if len(dt_tags) == len(dd_tags):
        for dt, dd in zip(dt_tags, dd_tags):
            a_tags = dt.find_all("a")
            [tag.decompose() for tag in a_tags] if a_tags else None
            yield from nested_handler(dt)
            yield "\n\n"
            yield from nested_handler(dd)
            yield "\n\n"
    else:
        # Case 2: 不匹配时，尝试顺序输出所有内容（保持原始顺序）
        for child in dl.children:
            if isinstance(child, Tag):
                if child.name == "dt":
                    yield from nested_handler(child)
                    yield "\nn"
                elif child.name == "dd":
                    yield from nested_handler(child)
                    yield "\nn"
                else:
                    yield from nested_handler(child)


def get_text(tag: Tag) -> Generator[str, None, None]:

    for child in tag.children:
        if isinstance(child, Doctype):
            continue
        if isinstance(child, NavigableString):
            yield child
        elif isinstance(child, Tag):
            if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                yield from get_title(child)
            elif child.name == "a":
                yield f"[{child.get_text(strip=False)}]({child.get('href')})"
            elif child.name == "img":
                yield f"![{child.get('alt', '')}]({child.get('src')})"
            elif child.name in ["strong", "b"]:
                yield f"**{child.get_text(strip=False)}**"
            elif child.name in ["em", "i"]:
                yield f"{child.get_text(strip=False)}"
            elif child.name == "br":
                yield "\n"
            elif child.name == "pre":
                grand_parent = child.parent.parent
                language = get_language(grand_parent)
                yield f"```{language}\n"
                yield from get_code(child)
                yield "\n```\n\n"
            elif child.name == "p":
                yield from get_text(child)
                yield "\n\n"
            elif child.name == "ul":
                yield from get_list(child, False, get_text)
            elif child.name == "ol":
                yield from get_list(child, True, get_text)
            elif child.name == "dl":
                yield from get_description(child, get_text)
            elif child.name == "div" and "dropdown" in child.attrs.get("class", [""]):
                yield child.get_text()
                yield "\n"
            elif child.name == "div" and "tabs-container" in child.attrs.get(
                "class", [""]
            ):
                tabs = child.find_all("li", {"role": "tab"})
                tab_panels = child.find_all("div", {"role": "tabpanel"})
                for tab, tab_panel in zip(tabs, tab_panels):
                    tab_name = tab.get_text(strip=True)
                    yield f"{tab_name}\n"
                    yield from get_text(tab_panel)
            elif child.name == "table":
                thead = child.find("thead")
                header_exists = isinstance(thead, Tag)
                if header_exists:
                    headers = thead.find_all("th")
                    if headers:
                        yield "| "
                        yield " | ".join(header.get_text() for header in headers)
                        yield " |\n"
                        yield "| "
                        yield " | ".join("----" for _ in headers)
                        yield " |\n"

                tbody = child.find("tbody")
                tbody_exists = isinstance(tbody, Tag)
                if tbody_exists:
                    for row in tbody.find_all("tr"):
                        yield "| "
                        yield " | ".join(
                            cell.get_text(strip=True).replace("\n", " ")
                            for cell in row.find_all("td")
                        )
                        yield " |\n"

                yield "\n\n"
            else:
                yield from get_text(child)


def langchain_recursive_url_extractor(raw_html, parser="html.parser") -> str:
    if isinstance(raw_html, BeautifulSoup):
        soup = raw_html
    else:
        soup = BeautifulSoup(raw_html, parser)
    # Remove all the tags that are not meaningful for the extraction.
    SCAPE_TAGS = ["nav", "footer", "aside", "script", "style", "button"]
    [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]
    article_element = soup.find("article")
    article_markdown = "".join(get_text(article_element))
    return re.sub(r"\n\n+", "\n\n", article_markdown).strip()


exclude_urls_ref = [
    "https://python.langchain.com/api_reference/_modules/",
    "https://python.langchain.com/api_reference/community/document_loaders/([^",
    "https://python.langchain.com/api_reference/langgraph_store_mongodb/api_docs.html",
    "https://python.langchain.com/api_reference/langgraph_checkpoint_mongodb/api_docs.html",
    "https://python.langchain.com/api_reference/_static/styles/theme.css?digest=8878045cc6db502f8baf",
    "https://python.langchain.com/api_reference/search.html",
    "https://python.langchain.com/api_reference/_static/copybutton.css?v=76b2166b",
    "https://python.langchain.com/api_reference/_static/scripts/pydata-sphinx-theme.js?digest=8878045cc6db502f8baf",
    "https://python.langchain.com/api_reference/_static/css/custom.css?v=8e9fa5b3",
    "https://python.langchain.com/api_reference/_static/scripts/bootstrap.js?digest=8878045cc6db502f8baf",
    "https://python.langchain.com/api_reference/_static/pygments.css?v=8f2a1f02",
    "https://python.langchain.com/api_reference/_static/styles/pydata-sphinx-theme.css?digest=8878045cc6db502f8baf",
    "https://python.langchain.com/api_reference/_static/sphinx-design.min.css?v=95c83b7e",
]

exclude_urls_doc = [
    "https://python.langchain.com/docs/integrations/huggingface",
    "https://python.langchain.com/docs/integrations/bedrock",
    "https://python.langchain.com/docs/integrations/openai",
    "https://python.langchain.com/docs/integrations/xai",
    "https://python.langchain.com/docs/integrations/ai21",
    "https://python.langchain.com/docs/concepts/few-shot-prompting",
    "https://python.langchain.com/docs/integrations/weaviate",
    "https://python.langchain.com/docs/integrations/tavily",
    "https://python.langchain.com/docs/integrations/wikipedia",
    "https://python.langchain.com/docs/integrations/google_vertex_ai_search",
    "https://python.langchain.com/docs/integrations/azure_ai_search",
    "https://python.langchain.com/docs/integrations/arxiv",
    "https://python.langchain.com/docs/integrations/elasticsearch_retriever",
    "https://python.langchain.com/docs/integrations/redis",
    "https://python.langchain.com/docs/integrations/faiss",
    "https://python.langchain.com/docs/integrations/retrievers/greennode-reranker",
    "https://python.langchain.com/docs/integrations/pinecone",
    "https://python.langchain.com/docs/integrations/milvus",
    "https://python.langchain.com/docs/use_cases/tool_use/quickstart",
    "https://python.langchain.com/docs/integrations/pdfplumber",
    "https://python.langchain.com/docs/integrations/tencent_cos_file",
    "https://python.langchain.com/docs/integrations/pymupdf4llm",
    "https://python.langchain.com/docs/integrations/microsoft_sharepoint",
    "https://python.langchain.com/docs/integrations/azure_blob_storage_file",
    "https://python.langchain.com/docs/integrations/slack",
    "https://python.langchain.com/docs/integrations/tencent_cos_directory",
    "https://python.langchain.com/docs/integrations/qdrant",
    "https://python.langchain.com/docs/integrations/json",
    "https://python.langchain.com/docs/integrations/huawei_obs_directory",
    "https://python.langchain.com/docs/integrations/bshtml",
    "https://python.langchain.com/docs/integrations/figma",
    "https://python.langchain.com/docs/integrations/aws_s3_file",
    "https://python.langchain.com/docs/integrations/recursive_url",
    "https://python.langchain.com/docs/integrations/clickhouse",
    "https://python.langchain.com/docs/integrations/pgvectorstore",
    "https://python.langchain.com/docs/integrations/databricks_vector_search",
    "https://python.langchain.com/docs/integrations/mongodb_atlas",
    "https://python.langchain.com/docs/integrations/elasticsearch",
    "https://python.langchain.com/docs/integrations/azure_openai",
    "https://python.langchain.com/docs/integrations/pgvector",
    "https://python.langchain.com/docs/integrations/chroma",
    "https://python.langchain.com/docs/integrations/couchbase",
    "https://python.langchain.com/docs/integrations/astradb",
    "https://python.langchain.com/docs/modules/agents/agent_types/react",
    "https://python.langchain.com/docs/modules/model_io/chat",
    "https://python.langchain.com/docs/integrations/openGauss",
    "https://python.langchain.com/docs/integrations/sqlserver",
    "https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever",
    "https://python.langchain.com/docs/expression_language/streaming",
    "https://python.langchain.com/docs/integrations/ollama",
    "https://python.langchain.com/docs/integrations/perplexity",
    "https://python.langchain.com/docs/integrations/ibm_watsonx",
    "https://python.langchain.com/docs/integrations/google_generative_ai",
    "https://python.langchain.com/docs/integrations/mistralai",
    "https://python.langchain.com/docs/integrations/llamacpp",
    "https://python.langchain.com/docs/integrations/together",
    "https://python.langchain.com/docs/integrations/nvidia_ai_endpoints",
    "https://python.langchain.com/docs/integrations/azure_chat_openai",
    "https://python.langchain.com/docs/integrations/anthropic",
    "https://python.langchain.com/docs/integrations/groq",
    "https://python.langchain.com/docs/integrations/fireworks",
    "https://python.langchain.com/docs/expression_language/how_to/decorator",
    "https://python.langchain.com/docs/integrations/databricks",
    "https://python.langchain.com/docs/integrations/google_vertex_ai_palm",
    "https://python.langchain.com/docs/integrations/cohere",
    "https://python.langchain.com/docs/expression_language/how_to/inspect",
    "https://python.langchain.com/docs/integrations/upstage",
    "https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent",
    "https://python.langchain.com/docs/expression_language/how_to/message_history",
    "https://python.langchain.com/docs/expression_language/how_to/configure",
    "https://python.langchain.com/docs/integrations/reddit",
    "https://python.langchain.com/docs/integrations/pypdfloader",
    "https://python.langchain.com/docs/integrations/docling",
    "https://python.langchain.com/docs/integrations/microsoft_onedrive",
    "https://python.langchain.com/docs/integrations/pypdfium2",
    "https://python.langchain.com/docs/api_reference/llms/langchain_gradient.llms.LangchainGradient",
    "https://python.langchain.com/docs/integrations/sitemap",
    "https://python.langchain.com/docs/integrations/providers/seekr",
    "https://python.langchain.com/docs/integrations/unstructured_file",
    "https://python.langchain.com/docs/integrations/facebook_chat",
    "https://python.langchain.com/docs/integrations/csv",
    "https://python.langchain.com/docs/integrations/hyperbrowser",
    "https://python.langchain.com/docs/integrations/google_cloud_storage_directory",
    "https://python.langchain.com/docs/integrations/quip",
    "https://python.langchain.com/docs/integrations/google_cloud_storage_file",
    "https://python.langchain.com/docs/integrations/pymupdf",
    "https://python.langchain.com/docs/integrations/pdfminer",
    "https://python.langchain.com/docs/integrations/google_drive",
    "https://python.langchain.com/docs/integrations/notion",
    "https://python.langchain.com/docs/integrations/github",
    "https://python.langchain.com/docs/integrations/azure_blob_storage_container",
    "https://python.langchain.com/docs/integrations/aws_s3_directory",
    "https://python.langchain.com/docs/integrations/web_base",
    "https://python.langchain.com/docs/integrations/trello",
    "https://python.langchain.com/docs/integrations/pypdfdirectory",
    "https://python.langchain.com/docs/integrations/whatsapp_chat",
    "https://python.langchain.com/docs/integrations/twitter",
    "https://python.langchain.com/docs/integrations/firecrawl",
    "https://python.langchain.com/docs/expression_language/cookbook/retrieval",
    "https://python.langchain.com/docs/api_reference/langchain-gradient_api_reference",
    "https://python.langchain.com/docs/integrations/vectorstores/Retrieval conceptual docs",
    "https://python.langchain.com/docs/integrations/mastodon",
    "https://python.langchain.com/docs/integrations/mathpix",
    "https://python.langchain.com/docs/integrations/discord",
    "https://python.langchain.com/docs/integrations/huawei_obs_file",
    "https://python.langchain.com/docs/integrations/dropbox",
    "https://python.langchain.com/docs/integrations/azure_ai_data",
    "https://python.langchain.com/docs/integrations/telegram",
    "https://python.langchain.com/docs/integrations/amazon_textract",
    "https://python.langchain.com/docs/integrations/roam",
    "https://python.langchain.com/docs/integrations/agentql",
    "https://python.langchain.com/docs/integrations/vectorstores/activeloop_deeplake",
    "https://python.langchain.com/docs/integrations/text_embedding/nomic",
]

exclude_urls_code = [
    "https://python.langchain.com/api_reference/_modules/langchain_community/document_loaders/([^",
]

if __name__ == "__main__":
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.pretty import pprint

    console = Console()
    url = "https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html#langchain_deepseek.chat_models.ChatDeepSeek"
    url = "https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.azure.AzureOpenAIEmbeddings.html"
    url = "https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html"
    url = "https://python.langchain.com/api_reference/tavily/tavily_search/langchain_tavily.tavily_search.TavilySearch.html#langchain_tavily.tavily_search.TavilySearch"
    # url = "https://python.langchain.com/docs/integrations/chat/"
    url = "https://python.langchain.com/docs/integrations/text_embedding/"
    # url = ""
    response = requests.get(url)

    metadata = langchain_recursive_url_metadata_extractor(
        raw_html=response.text,
        url=url,
        response=response,
        type="doc",
        lang="python",
    )

    pprint(metadata)
    doc = langchain_recursive_url_extractor(response.text, parser="lxml")
    md = Markdown(doc)
    console.print(md)

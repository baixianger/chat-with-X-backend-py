"""
Parser for the langgraph recursive url.
"""
# pylint: disable=wrong-import-position
# pylint: disable=line-too-long
# pylint: disable=unused-argument
# pylint: disable=expression-not-assigned
# type: ignore
import os
import sys
import re
from typing import Generator, Callable, Union
import aiohttp
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag, Doctype

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def langgraph_recursive_url_metadata_extractor(
    raw_html: str,
    url: str,
    response: Union[requests.Response, aiohttp.ClientResponse],
    **kwargs,
) -> dict:
    """Extract metadata from the langgraph recursive url."""
    soup = BeautifulSoup(raw_html, "lxml")
    title_element = soup.find("title")
    try:
        title_element.find("a").decompose() # type: ignore
    except AttributeError:
        pass
    title = (
        title_element.get_text() if title_element else url.rstrip("/").split("/")[-1]
    )
    return {
        "source": url,
        "title": title,
        **kwargs,
    }


def get_title(title: Tag) -> Generator[str, None, None]:
    """Get the title of the tag."""
    a_tag = title.find("a")
    if a_tag:
        a_tag.decompose()
    yield f"{'#' * int(title.name[1:])} {title.get_text(strip=True)}\n\n"

def get_code_line(code: Tag):
    """Get the code line from the code tag."""
    for child in code.children:
        if child.name == "span": # type: ignore
            yield child.get_text()
        else:
            continue


def get_list(
    list_tag: Tag,
    ordered: bool,
    nested_handler: Callable[[Tag], Generator[str, None, None]],
) -> Generator[str, None, None]:
    """Get the list from the list tag."""
    for i, li in enumerate(list_tag.find_all("li", recursive=False)):
        prefix = f"{i + 1}. " if ordered else "- "
        yield prefix
        yield from nested_handler(li) # type: ignore
        yield "\n\n"


def get_table(table: Tag) -> Generator[str, None, None]:
    """Get the table from the table tag."""
    thead = table.find("thead")
    if isinstance(thead, Tag):
        headers = thead.find_all("th")
        if headers:
            yield "| " + " | ".join(header.get_text() for header in headers) + " |\n"
            yield "| " + " | ".join("----" for _ in headers) + " |\n"

    tbody = table.find("tbody")
    if isinstance(tbody, Tag):
        for row in tbody.find_all("tr"):
            yield "| " + " | ".join(
                cell.get_text(strip=True).replace("\n", " ")
                for cell in row.find_all("td") # type: ignore
            ) + " |\n"

# type: ignore
def get_toc(nav: Tag, level: int = 0) -> Generator[str, None, None]:
    """Get the table of contents from the nav tag."""
    for child in nav.children:
        if child.name == "label": # type: ignore
            yield child.get_text(strip=True)
        elif child.name == "ul": # type: ignore
            for li in child.find_all("li", recursive=False): # type: ignore
                a_tag = li.find("a", recursive=False) # type: ignore
                if a_tag and a_tag.get("href"):
                    title = a_tag.get_text(strip=True)

                    # 识别 code class 中的 doc-symbol 类型
                    code = a_tag.find("code")
                    prefix = ""
                    if code and "doc-symbol-function" in code.get("class", []):
                        prefix = "func "
                    elif code and "doc-symbol-class" in code.get("class", []):
                        prefix = "class "
                    elif code and "doc-symbol-method" in code.get("class", []):
                        prefix = "meth "
                    elif code and "doc-symbol-attribute" in code.get("class", []):
                        prefix = "attr "

                    url = a_tag["href"]
                    indent = "  " * level
                    yield f"{indent}- [{prefix}{title}]({url})"
                if li.find("nav", recursive=False):
                    sub_nav = li.find("nav", recursive=False)
                    yield from get_toc(sub_nav, level + 1)
        else:
            continue

SCAPE_TAGS = ["footer", "aside", "script", "style"]


def langgraph_recursive_url_extractor(raw_html: str) -> str:
    """Extract the text from the raw html."""
    if isinstance(raw_html, BeautifulSoup):
        soup = raw_html
    else:
        soup = BeautifulSoup(raw_html, "lxml")
    # Remove all the tags that are not meaningful for the extraction.

    [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]

    # find table of content nav
    toc = soup.find("nav", {"aria-label": "Table of contents"})  # type: ignore
    if isinstance(toc, Tag):
        table_of_content = "\n".join(get_toc(toc))
    else:
        table_of_content = ""

    article_element = soup.find("article")

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
                    yield f"_{child.get_text(strip=False)}_"
                elif child.name == "br":
                    yield "\n"
                elif child.name == "code":
                    parent = child.find_parent()
                    if parent is not None and parent.name == "pre": # type: ignore
                        classes = parent.parent.attrs.get("class", "") # type: ignore

                        language = next(
                            filter(lambda x: re.match(r"language-\w+", x), classes),
                            None,
                        )
                        if language is None:
                            language = ""
                        else:
                            language = language.split("-")[1]

                        # code_content = "".join(get_code_line(child))
                        code_content = child.get_text()
                        yield f"```{language}\n{code_content}\n```\n\n"
                    else:
                        yield f"`{child.get_text(strip=False)}`"

                elif child.name == "p":
                    yield from get_text(child)
                    yield "\n\n"
                elif child.name == "ul":
                    yield from get_list(child, ordered=False, nested_handler=get_text)
                elif child.name == "ol":
                    yield from get_list(child, ordered=True, nested_handler=get_text)
                elif child.name == "div" and "tabs-container" in child.attrs.get(
                    "class", [""]
                ):
                    tabs = child.find_all("li", {"role": "tab"})
                    tab_panels = child.find_all("div", {"role": "tabpanel"})
                    for tab, tab_panel in zip(tabs, tab_panels):
                        tab_name = tab.get_text(strip=True)
                        yield f"{tab_name}\n"
                        yield from get_text(tab_panel) # type: ignore
                elif child.name == "table":
                    yield from get_table(child)
                elif child.name in ["button"]:
                    continue
                else:
                    yield from get_text(child)
    if isinstance(article_element, Tag):
        article_content = "".join(get_text(article_element))
    else:
        article_content = ""
    md_content = f"{table_of_content}\n\n{article_content}"
    return re.sub(r"\n\n+", "\n\n", md_content).strip()


if __name__ == "__main__":

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.pretty import pprint

    console = Console()
    TEST_URL = "https://langchain-ai.github.io/langgraph/reference/graphs/"

    test_response = requests.get(TEST_URL, timeout=10)
    metadata = langgraph_recursive_url_metadata_extractor(
        raw_html=test_response.text,
        url=TEST_URL,
        response=test_response,
        doc_type="doc",
        lang="python",
    )
    pprint(metadata)
    doc = langgraph_recursive_url_extractor(test_response.text)
    md = Markdown(doc)
    console.print(md)

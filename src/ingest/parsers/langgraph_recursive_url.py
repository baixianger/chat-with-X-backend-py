import os
import sys
import re
import requests
import aiohttp
from typing import Generator, Optional, Union, Literal, Callable
from bs4 import BeautifulSoup, Doctype, NavigableString, Tag

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def langgraph_recursive_url_metadata_extractor(
    raw_html: str,
    url: str,
    response: Union[requests.Response, aiohttp.ClientResponse],
    *,
    type: Literal["documents", "api_reference", "source_code"],
    lang: Optional[str] = "python",
) -> dict:
    soup = BeautifulSoup(raw_html, "lxml")
    title_element = soup.find("title")
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
    if a_tag and a_tag.get("href"):
        url = a_tag["href"]
        yield f"{'#' * int(title.name[1:])} [{title.get_text(strip=True)[:-2]}]({url})\n\n"
    else:
        yield f"{'#' * int(title.name[1:])} {title.get_text(strip=True)}\n\n"


def get_code_line(code: Tag):
    for child in code.children:
        if child.name == "span":
            yield child.get_text()
        else:
            continue


def get_list(
    list: Tag,
    ordered: bool,
    nested_handler: Callable[[Tag], Generator[str, None, None]],
) -> Generator[str, None, None]:
    for i, li in enumerate(list.find_all("li", recursive=False)):
        prefix = f"{i + 1}. " if ordered else "- "
        yield prefix
        yield from nested_handler(li)
        yield "\n\n"


def get_table(table: Tag) -> Generator[str, None, None]:
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
                for cell in row.find_all("td")
            ) + " |\n"


def get_toc(nav: Tag, level: int = 0) -> Generator[str, None, None]:
    # TOC 标题
    for child in nav.children:
        if child.name == "label":
            yield child.get_text(strip=True)
        elif child.name == "ul":
            for li in child.find_all("li", recursive=False):
                a_tag = li.find("a", recursive=False)
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


def langgraph_recursive_url_extractor(soup: BeautifulSoup) -> str:
    # Remove all the tags that are not meaningful for the extraction.
    SCAPE_TAGS = ["footer", "aside", "script", "style"]
    [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]

    # find table of content nav
    toc = soup.find("nav", {"aria-label": "Table of contents"})
    if toc:
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
                    if parent is not None and parent.name == "pre":
                        classes = parent.parent.attrs.get("class", "")

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
                        yield from get_text(tab_panel)
                elif child.name == "table":
                    yield from get_table(child)
                elif child.name in ["button"]:
                    continue
                else:
                    yield from get_text(child)

    article_content = "".join(get_text(article_element))
    md_content = f"{table_of_content}\n\n{article_content}"
    return re.sub(r"\n\n+", "\n\n", md_content).strip()


if __name__ == "__main__":

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.pretty import pprint

    console = Console()
    url = "https://langchain-ai.github.io/langgraph/reference/graphs/"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    metadata = langgraph_recursive_url_metadata_extractor(
        raw_html=response.text,
        url=url,
        response=response,
        type="api",
        lang="python",
    )
    pprint(metadata)
    doc = langgraph_recursive_url_extractor(soup)
    md = Markdown(doc)
    console.print(md)

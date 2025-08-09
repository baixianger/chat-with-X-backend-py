"""
Parser for the langsmith recursive url.
"""
# pylint: disable=wrong-import-position
# pylint: disable=line-too-long
# pylint: disable=unused-argument
# pylint: disable=expression-not-assigned
# type: ignore
import os
import sys
import re
from typing import Generator, Optional, Union, Literal
import requests
import aiohttp
from bs4 import BeautifulSoup
from bs4.element import Doctype, NavigableString, Tag, AttributeValueList

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def langsmith_recursive_url_metadata_extractor(
    raw_html: str,
    url: str,
    response: Union[requests.Response, aiohttp.ClientResponse],
    **kwargs,
) -> dict:
    """Extract metadata from the langsmith recursive url."""
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


def get_language(div: Tag) -> str:
    """Get the language of the div tag."""
    # highlight-<language> class is for api reference pages
    # language-<language> class is for documents pages
    classes = div.get("class", AttributeValueList())
    if classes is None:
        return ""
    for cls in classes:
        if re.match(r"(highlight|language)-\w+", cls):
            return cls.split("-")[1]
    return ""


def get_code(pre: Tag) -> str:
    """Get the code from the pre tag."""
    if pre.find("code"):
        lines = []
        code = pre.find("code")
        if isinstance(code, Tag):
            for child in code.children:
                lines.append(child.get_text())
            return "\n".join(lines) + "\n"

    for a_tag in pre.find_all("a"):
        a_tag.decompose()  # for source code pages, there are links in the code block
    return pre.get_text()


SCAPE_TAGS = ["nav", "footer", "aside", "script", "style", "button"]

def langsmith_recursive_url_extractor(soup: BeautifulSoup) -> str:
    """Extract the text from the raw html."""
    # Remove all the tags that are not meaningful for the extraction.
    [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]
    article_element = soup.find("article")

    if not isinstance(article_element, Tag):
        return ""


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
                elif child.name == "dt":
                    yield child.get_text(strip=False)
                    yield "\n"
                elif child.name == "pre":
                    grand_parent = child.parent.parent # type: ignore
                    language = get_language(grand_parent) # type: ignore
                    code_content = get_code(child)
                    yield f"```{language}\n{code_content}\n```\n\n"
                elif child.name == "p":
                    yield from get_text(child)
                    yield "\n\n"
                elif child.name == "ul":
                    for li in child.find_all("li", recursive=False):
                        yield "- "
                        yield from get_text(li) # type: ignore
                        yield "\n\n"
                elif child.name == "ol":
                    for i, li in enumerate(child.find_all("li", recursive=False)):
                        yield f"{i + 1}. "
                        yield from get_text(li) # type:ignore
                        yield "\n\n"
                elif child.name == "div" and "dropdown" in child.attrs.get(
                    "class", [""]
                ):
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
                        yield from get_text(tab_panel) # type: ignore
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
                                for cell in row.find_all("td") # type: ignore
                            )
                            yield " |\n"

                    yield "\n\n"
                elif child.name in ["button"]:
                    continue
                else:
                    yield from get_text(child)

    article_content = "".join(get_text(article_element))
    return re.sub(r"\n\n+", "\n\n", article_content).strip()


if __name__ == "__main__":
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.pretty import pprint

    console = Console()

    def test():
        """test doc parser"""
        # doc
        url = "https://docs.smith.langchain.com/observability/concepts/"
        # ref
        url = "https://docs.smith.langchain.com/reference/python/client/langsmith.client.Client"
        # code
        url = "https://docs.smith.langchain.com/reference/python/_modules/langsmith/client"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")
        metadata = langsmith_recursive_url_metadata_extractor(
            raw_html=response.text,
            url=url,
            response=response,
            type="code",
            lang="python",
        )
        pprint(metadata)
        doc = langsmith_recursive_url_extractor(soup)
        md = Markdown(doc)
        console.print(md)

    test()

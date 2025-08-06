import os
import sys
import re
import requests
import aiohttp
from typing import Generator, Optional, Union, Literal
from bs4 import BeautifulSoup, Doctype, NavigableString, Tag

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def langsmith_recursive_url_metadata_extractor(
    raw_html: str,
    url: str,
    response: Union[requests.Response, aiohttp.ClientResponse],
    *,
    type: Literal["documents", "api_reference", "source_code"],
    lang: Optional[str] = "",
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
        "lang": lang,
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
    if pre.find("code"):
        lines = []
        code = pre.find("code")
        for child in code.children:
            lines.append(child.get_text())
        return "\n".join(lines) + "\n"

    for a_tag in pre.find_all("a"):
        a_tag.decompose()  # for source code pages, there are links in the code block
    return pre.get_text()


def langsmith_recursive_url_extractor(soup: BeautifulSoup) -> str:
    # Remove all the tags that are not meaningful for the extraction.
    SCAPE_TAGS = ["nav", "footer", "aside", "script", "style", "button"]
    [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]
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
                elif child.name == "dt":
                    yield child.get_text(strip=False)
                    yield "\n"
                elif child.name == "pre":
                    grand_parent = child.parent.parent
                    language = get_language(grand_parent)
                    code_content = get_code(child)
                    yield f"```{language}\n{code_content}\n```\n\n"
                elif child.name == "p":
                    yield from get_text(child)
                    yield "\n\n"
                elif child.name == "ul":
                    for li in child.find_all("li", recursive=False):
                        yield "- "
                        yield from get_text(li)
                        yield "\n\n"
                elif child.name == "ol":
                    for i, li in enumerate(child.find_all("li", recursive=False)):
                        yield f"{i + 1}. "
                        yield from get_text(li)
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

import os
import sys
import re
from typing import Optional, Literal
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def sitemap_meta_extractor(  # only work for sitemap loader
    meta: dict,
    soup: BeautifulSoup,
    *,
    type: Literal["documents", "api_reference", "source_code"],
    lang: Optional[str] = None,
) -> dict:
    title_element = soup.find("h1")
    title = (
        title_element.get_text()
        if title_element
        else meta["loc"].rstrip("/").split("/")[-1]
    )
    return {
        "source": meta[
            "loc"
        ],  # sitemaploader will use loc as the key to identify the link, but we use it as the source
        "title": title,
        "type": type,
        "lang": lang if lang else "",
    }


def site_map_parsing_function(soup: BeautifulSoup) -> str:
    """SitemapLoader's parsing function only accept BeautifulSoup object"""
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

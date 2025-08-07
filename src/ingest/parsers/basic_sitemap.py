import os
import sys
import re
from typing import Optional, Literal
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def sitemap_meta_extractor(  # only work for sitemap loader
    meta: dict,
    soup: BeautifulSoup,
    doc_type: Optional[Literal["doc", "ref", "code"]] = None,
    lang: Optional[str] = None,
) -> dict:
    """Extract metadata from the sitemap."""
    title_element = soup.find("h1")
    title = (
        title_element.get_text()
        if title_element
        else meta["loc"].rstrip("/").split("/")[-1]
    )
    return {
        "source": meta[
            "loc"
        ],
        "title": title,
        "type": doc_type,
        "lang": lang if lang else "",
    }


def site_map_parsing_function(soup: BeautifulSoup) -> str:
    """SitemapLoader's parsing function only accept BeautifulSoup object"""
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

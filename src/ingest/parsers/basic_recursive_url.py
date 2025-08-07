"""Basic recursive url metadata extractor and extractor."""
# pylint: disable=unused-argument
import os
import sys
import re
from typing import Optional, Union, Literal
import requests
import aiohttp
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def recursive_url_metadata_extractor(
    raw_html: str,
    url: str,
    response: Union[requests.Response, aiohttp.ClientResponse],
    **kwargs,
) -> dict:
    """Extract metadata from the recursive url."""
    soup = BeautifulSoup(raw_html, "lxml")
    title_element = soup.find("h1")
    title = (
        title_element.get_text() if title_element else url.rstrip("/").split("/")[-1]
    )
    return {
        "source": url,
        "title": title,
        **kwargs,
    }


def recursive_url_extractor(raw_html: str) -> str:
    """RecursiveUrlLoader's parsing function only accept raw html text from request response"""
    soup = BeautifulSoup(raw_html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

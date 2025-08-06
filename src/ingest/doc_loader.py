import os
import sys
import requests
import aiohttp
from typing import Optional, Callable, Union, Literal, Any
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.ingest.parsers.basic_sitemap import (
    sitemap_meta_extractor,
    site_map_parsing_function,
)
from src.ingest.parsers.basic_recursive_url import (
    recursive_url_metadata_extractor,
    recursive_url_extractor,
)


# ==================
# SitemapLoader
# ==================


def site_map_loader(
    path: str,
    filter_urls: Optional[list[str]] = None,
    meta_function: Optional[
        Callable[[dict[str, Any], BeautifulSoup], dict[str, Any]]
    ] = sitemap_meta_extractor,
    parsing_function: Optional[
        Callable[[BeautifulSoup], str]
    ] = site_map_parsing_function,
    default_parser: Literal["lxml", "html.parser"] = "lxml",
    bs_kwargs: Optional[dict[str, Any]] = None,
    meta_kwargs: Optional[dict[str, Any]] = None,
):
    meta_function = lambda meta, soup: meta_function(meta, soup, **meta_kwargs)
    loader = SitemapLoader(
        web_path=path,
        meta_function=meta_function,
        bs_kwargs=bs_kwargs,
        filter_urls=filter_urls,
        parsing_function=parsing_function,
        default_parser=default_parser,
    )
    return loader.load()


# ==================
# RecursiveUrlLoader
# ==================


def recursive_url_loader(
    path: str,
    filter_urls: Optional[list[str]] = None,
    metadata_extractor: Optional[
        Callable[
            [str, str, Union[requests.Response, aiohttp.ClientResponse]], dict[str, Any]
        ]
    ] = recursive_url_metadata_extractor,
    extractor: Optional[Callable[[str], str]] = recursive_url_extractor,
    meta_kwargs: Optional[dict[str, Any]] = None,
    max_depth: Optional[int] = 5,
):
    loader = RecursiveUrlLoader(
        url=path,
        max_depth=max_depth,
        metadata_extractor=lambda raw_html, url, response: metadata_extractor(
            raw_html, url, response, **meta_kwargs
        ),
        extractor=extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=filter_urls,
    )
    return loader.load()


if __name__ == "__main__":

    meta_kwargs = {"type": "documents", "lang": "python"}
    langchain_doc = recursive_url_loader(
        path="https://python.langchain.com/docs/",
        max_depth=4,
        meta_kwargs=meta_kwargs,
    )
    # save doc in txt file
    for doc in langchain_doc:
        with open(f"./data/{doc.metadata['title']}.txt", "w") as f:
            f.write(doc.page_content)

import logging

logger = logging.getLogger(__name__)
from ddgs import DDGS
from src.controller.tools.Tool import Tool
from pydantic import BaseModel


class WebSearchInput(BaseModel):
    query: str
    max_results: int = 5


class WebSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="A comprehensive web search engine. Use this tool to retrieve real-time information, verify facts, or find details on topics outside your training data.",
            schema=WebSearchInput,
            func=web_search,
        )


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of search results with title, link, and snippet
    """
    logger.info(f"called web search with query: {query}")
    try:
        results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": result.get("title"),
                        "link": result.get("href"),
                        "snippet": result.get("body"),
                    }
                )
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

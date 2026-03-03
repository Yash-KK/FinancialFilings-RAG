from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool




wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list", backend="news")

@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time or current information using Tavily.

    **Use this for LIVE or RECENT information not in your training data.**
    **For historical financial documents (SEC filings) → use hybrid_search instead.**

    Use this for queries like:
    - "Latest news on Apple"
    - "Current stock price of TCS"
    - "Recent earnings announcement for Microsoft"
    - "What happened in the market today?"

    Do NOT use this for:
    - Historical SEC filings or earnings reports → use hybrid_search

    Args:
        query: Natural language search query
    """
    try:
        results = search.invoke(query)
        if not results:
            return f"No results found for: '{query}'"
        return f"[WEB SEARCH RESULTS]\nQuery: {query}\n\n{results}"
    except Exception as e:
        return f"Error during web search for '{query}': {str(e)}"
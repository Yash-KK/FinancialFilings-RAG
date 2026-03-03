from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool




wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list", backend="news")

@tool
def web_search(query: str) -> str:
    """
    Search the web for LIVE, REAL-TIME, or RECENT information not available in SEC filings.

    **IMPORTANT: This is the FALLBACK tool. Always try hybrid_search FIRST.**
    **Only use this tool when:**
    - hybrid_search returns no results or insufficient data
    - User explicitly asks for "current", "live", "real-time", or "latest" information
    - Query is about live stock prices, today's market movement, or breaking news

    This tool searches the live web and retrieves:
    - Current stock prices and market data
    - Recent news headlines and articles
    - Analyst ratings and price targets published recently
    - Earnings announcements and press releases not yet in SEC filings
    - Macroeconomic news (Fed decisions, CPI, interest rates)

    Use this for queries like:
    - "What is Apple's stock price right now?"
    - "Latest news on TCS earnings"
    - "What did the Fed announce today?"
    - "Recent analyst upgrades for Microsoft"
    - "What happened to Reliance stock this week?"

    Do NOT use this for:
    - Historical revenue, profit, or EPS data → use hybrid_search
    - SEC filings (10-K, 10-Q, 8-K) → use hybrid_search
    - Year-over-year or quarter-over-quarter comparisons → use hybrid_search
    - Any financial data older than a few weeks → use hybrid_search

    Returns:
        Web search results with title, snippet, and source URL for each result.
        Always cite the source URL in your final response.

    Args:
        query: Natural language search query e.g. "Apple stock price today" or "TCS Q3 2025 earnings news"
    """
    try:
        results = search.invoke(query)
        if not results:
            return f"No results found for: '{query}'. Try rephrasing or use hybrid_search for historical data."
        return f"[WEB SEARCH RESULTS]\nQuery: {query}\n\n{results}"
    except Exception as e:
        return f"Error during web search for '{query}': {str(e)}"
import re
import yfinance as yf
from langchain.tools import tool


def _resolve_ticker(ticker: str, exchange: str = "NS") -> str:
    ticker = ticker.upper().strip()
    if "." not in ticker and ticker.isalpha():
        return f"{ticker}.{exchange}"
    return ticker


def _format_number(value, prefix="$") -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if value >= 1_000_000_000_000:
            return f"{prefix}{value / 1_000_000_000_000:.2f}T"
        elif value >= 1_000_000_000:
            return f"{prefix}{value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{prefix}{value / 1_000_000:.2f}M"
        return f"{prefix}{value:,.2f}"
    except (TypeError, ValueError):
        return "N/A"

_STOPWORDS = {
    "WHAT", "IS", "THE", "OF", "AT", "IN", "FOR", "GET", "ME", "GIVE",
    "SHOW", "TELL", "HOW", "HAS", "ARE", "WAS", "WERE", "WILL", "DOES",
    "STOCK", "PRICE", "SHARE", "MARKET", "LATEST", "RECENT", "CURRENT",
    "LIVE", "TODAY", "NOW", "TRADING", "NEWS", "HEADLINE", "ARTICLE",
    "PERFORMANCE", "HISTORY", "TREND", "RETURN", "CHANGE", "LAST", "PAST",
    "OVER", "THIS", "YEAR", "MONTH", "WEEK", "DAY", "AND", "WITH", "FROM",
}


def _extract_intent(query: str) -> dict:
    q = query.lower()

    # --- Detect mode ---
    # Be specific: "latest news", "recent news", not just "latest"
    news_phrases = ["latest news", "recent news", "news on", "news about",
                    "headlines", "articles about", "what's happening"]
    history_phrases = ["history", "performance", "trend", "return",
                       "how has", "how did", "change", "chart", "moved"]

    if any(phrase in q for phrase in news_phrases):
        mode = "news"
    elif any(phrase in q for phrase in history_phrases):
        mode = "history"
    else:
        mode = "price"  # default — covers "price", "latest price", "trading at", etc.

    # --- Detect period ---
    period_map = {
        "1d":  ["1d", "today", "1 day"],
        "5d":  ["5d", "5 day", "week"],
        "1mo": ["1mo", "1 month", "this month", "monthly"],
        "3mo": ["3mo", "3 month", "quarter"],
        "6mo": ["6mo", "6 month"],
        "1y":  ["1y", "1 year", "annual", "yearly", "this year"],
        "2y":  ["2y", "2 year"],
        "5y":  ["5y", "5 year"],
    }
    period = "1mo"
    for p, keywords in period_map.items():
        if any(kw in q for kw in keywords):
            period = p
            break

    # --- Detect exchange ---
    exchange = "BO" if ".bo" in q or "bse" in q else "NS"

    # --- Detect limit for news ---
    limit_match = re.search(r"(\d+)\s*(news|article|headline)", q)
    limit = int(limit_match.group(1)) if limit_match else 5
    limit = min(max(limit, 1), 10)

    # --- Extract ticker: uppercase words excluding stopwords ---
    # First check for explicit ticker with suffix e.g. TCS.NS, RELIANCE.BO
    explicit_match = re.search(r"\b([A-Z]{1,10}\.[A-Z]{1,3})\b", query.upper())
    if explicit_match:
        ticker = explicit_match.group(1)
    else:
        # Find all candidate uppercase tokens and skip stopwords
        candidates = re.findall(r"\b([A-Z]{2,10})\b", query.upper())
        ticker = next((c for c in candidates if c not in _STOPWORDS), None)

    return {
        "ticker": ticker,
        "mode": mode,
        "period": period,
        "exchange": exchange,
        "limit": limit,
    }

def _get_price(stock, resolved):
    info = stock.info
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    if price is None:
        return f"No live data found for '{resolved}'. Verify the ticker or try exchange suffix .BO."

    currency = info.get("currency", "")
    currency_symbol = "$" if currency == "USD" else f"{currency} "

    return (
        f"[LIVE MARKET DATA — Yahoo Finance]\n"
        f"Ticker:         {resolved}\n"
        f"Company:        {info.get('longName', 'N/A')}\n"
        f"Exchange:       {info.get('exchange', 'N/A')} ({currency})\n"
        f"Current Price:  {currency_symbol}{price:,.2f}\n"
        f"Market Cap:     {_format_number(info.get('marketCap'), currency_symbol)}\n"
        f"52W High:       {currency_symbol}{info.get('fiftyTwoWeekHigh', 'N/A')}\n"
        f"52W Low:        {currency_symbol}{info.get('fiftyTwoWeekLow', 'N/A')}\n"
        f"P/E Ratio:      {info.get('trailingPE', 'N/A')}\n"
        f"Dividend Yield: {info.get('dividendYield', 'N/A')}\n"
        f"Sector:         {info.get('sector', 'N/A')}\n"
        f"Industry:       {info.get('industry', 'N/A')}"
    )


def _get_history(stock, resolved, period):
    hist = stock.history(period=period)

    if hist.empty:
        return f"No price history found for '{resolved}' over '{period}'."

    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]
    change_abs = end_price - start_price
    change_pct = (change_abs / start_price) * 100
    direction = "▲" if change_pct >= 0 else "▼"

    return (
        f"[PRICE HISTORY — Yahoo Finance]\n"
        f"Ticker:       {resolved} | Period: {period}\n"
        f"Start Price:  ${start_price:.2f}\n"
        f"End Price:    ${end_price:.2f}\n"
        f"Change:       {direction} ${abs(change_abs):.2f} ({change_pct:+.2f}%)\n"
        f"Period High:  ${hist['High'].max():.2f}\n"
        f"Period Low:   ${hist['Low'].min():.2f}\n"
        f"Avg Volume:   {int(hist['Volume'].mean()):,}\n"
        f"Trading Days: {len(hist)}"
    )


def _get_news(stock, resolved, limit):
    news = stock.news[:limit]

    if not news:
        return f"No recent news found for '{resolved}'."

    headlines = []
    for i, article in enumerate(news, 1):
        content = article.get("content", {})
        title = content.get("title", "No title")
        provider = content.get("provider", {}).get("displayName", "Unknown source")
        headlines.append(f"{i}. [{provider}] {title}")

    return f"[LATEST NEWS — Yahoo Finance]\nTicker: {resolved}\n\n" + "\n".join(headlines)


@tool
def live_finance_searcher(query: str) -> str:
    """
    Fetch LIVE, REAL-TIME financial data from Yahoo Finance using a natural language query.

    **IMPORTANT: Use this tool ONLY for current/live market data.**
    **For historical financials from SEC filings → use hybrid_search instead.**

    Automatically detects:
    - Ticker symbol from the query
    - Mode: 'price' (default), 'history', or 'news'
    - Period for history queries (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
    - Exchange: NSE (default) or BSE for Indian stocks

    Use this for queries like:
    - "TCS stock price"
    - "Apple stock performance this year"
    - "Latest 5 news articles on Infosys"
    - "RELIANCE.BO 6 month history"
    - "What is Microsoft trading at?"

    Do NOT use this for:
    - Historical revenue, profit, EPS → use hybrid_search
    - SEC filings (10-K, 10-Q, 8-K) → use hybrid_search
    - Year-over-year financial comparisons → use hybrid_search

    Args:
        query: Natural language query e.g. "TCS stock price" or "Apple news" or "Infosys 1 year history"
    """
    intent = _extract_intent(query)
    ticker = intent["ticker"]

    if not ticker:
        return (
            f"Could not extract a ticker symbol from query: '{query}'. "
            f"Please include the stock symbol e.g. 'TCS stock price' or 'AAPL news'."
        )

    try:
        resolved = _resolve_ticker(ticker, intent["exchange"])
        stock = yf.Ticker(resolved)

        if intent["mode"] == "price":
            return _get_price(stock, resolved)
        elif intent["mode"] == "history":
            return _get_history(stock, resolved, intent["period"])
        elif intent["mode"] == "news":
            return _get_news(stock, resolved, intent["limit"])

    except Exception as e:
        return (
            f"Error processing query '{query}' (ticker={ticker}): {str(e)}. "
            f"For Indian stocks, try appending .NS or .BO e.g. 'TCS.NS price'."
        )
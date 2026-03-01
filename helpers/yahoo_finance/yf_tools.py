import yfinance as yf
from langchain.tools import tool
from yahoo_finance.schema import StockHistoryInput

INDIAN_SUFFIXES = {".NS", ".BO"}


def _resolve_ticker(ticker: str, exchange: str = "NS") -> str:
    """Auto-append exchange suffix for Indian tickers."""
    ticker = ticker.upper().strip()
    if "." not in ticker and ticker.isalpha():
        return f"{ticker}.{exchange}"
    return ticker


def _format_number(value, prefix="$") -> str:
    """Format large numbers into human-readable form."""
    if value is None or value == "N/A":
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


@tool
def get_stock_price(ticker: str, exchange: str = "NS") -> str:
    """
    Get the CURRENT, LIVE stock price and key market data for a given ticker symbol via Yahoo Finance.

    **IMPORTANT: Use this tool for REAL-TIME or CURRENT market data ONLY.**
    **For historical financials, revenue, earnings from SEC filings → use hybrid_search instead.**

    This tool retrieves:
    - Live/current stock price
    - Market capitalization
    - 52-week high and low
    - P/E ratio, dividend yield
    - Currency and exchange info

    Use this for queries like:
    - "What is Apple's stock price today?"
    - "What is TCS trading at right now?"
    - "Current market cap of Reliance?"

    Do NOT use this for:
    - Historical revenue or earnings → use hybrid_search
    - Past quarterly results → use hybrid_search
    - Year-over-year comparisons → use hybrid_search

    Args:
        ticker: Stock ticker symbol e.g. AAPL, TCS, INFY, RELIANCE
        exchange: Exchange suffix for Indian stocks. 'NS' for NSE (default), 'BO' for BSE.
                  Ignored for US/international tickers that already have a suffix.
    """
    try:
        resolved = _resolve_ticker(ticker, exchange)
        stock = yf.Ticker(resolved)
        info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            return (
                f"No live data found for '{resolved}'. "
                f"Verify the ticker is correct. For Indian stocks, try exchange='NS' or 'BO'."
            )

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
    except Exception as e:
        return (
            f"Error fetching live price for '{ticker}': {str(e)}. "
            f"For Indian stocks, ensure correct suffix — try TCS.NS or TCS.BO."
        )


@tool(args_schema=StockHistoryInput)
def get_stock_history(ticker: str, period: str = "1mo", exchange: str = "NS") -> str:
    """
    Get recent stock price performance and trading history from Yahoo Finance.

    **Use this for PRICE MOVEMENT and TRADING DATA over a time window.**
    **For fundamental financial data (revenue, profit, EPS) → use hybrid_search instead.**

    This tool retrieves:
    - Price change over a selected period (open → close)
    - Percentage gain/loss
    - Period high, low, and average trading volume

    Supported periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y

    Use this for queries like:
    - "How has TCS stock performed this month?"
    - "What's Apple's 6-month price trend?"
    - "Has Infosys gone up or down this year?"

    Do NOT use this for:
    - Revenue or earnings comparisons → use hybrid_search
    - SEC filing data → use hybrid_search

    Args:
        ticker: Stock ticker symbol e.g. AAPL, TCS, INFY
        period: Time window for history. One of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y (default: 1mo)
        exchange: Exchange suffix for Indian stocks. 'NS' for NSE (default), 'BO' for BSE.
    """
    try:
        resolved = _resolve_ticker(ticker, exchange)
        stock = yf.Ticker(resolved)
        hist = stock.history(period=period)

        if hist.empty:
            return (
                f"No historical data found for '{resolved}' over period '{period}'. "
                f"Check the ticker or try a different exchange suffix."
            )

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
    except Exception as e:
        return f"Error fetching price history for '{ticker}': {str(e)}"


@tool
def get_stock_news(ticker: str, limit: int = 5, exchange: str = "NS") -> str:
    """
    Get the latest news headlines for a stock from Yahoo Finance.

    **Use this for RECENT NEWS and MARKET SENTIMENT.**
    **For earnings, filings, or financial reports → use hybrid_search instead.**

    This tool retrieves:
    - Recent news headlines from Yahoo Finance
    - Publication source and summary

    Use this for queries like:
    - "Any recent news on Infosys?"
    - "What's happening with Tesla lately?"
    - "Latest headlines for Reliance?"

    Args:
        ticker: Stock ticker symbol e.g. AAPL, TCS, INFY
        limit: Number of news articles to return (default: 5, max: 10)
        exchange: Exchange suffix for Indian stocks. 'NS' for NSE (default), 'BO' for BSE.
    """
    try:
        resolved = _resolve_ticker(ticker, exchange)
        stock = yf.Ticker(resolved)
        news = stock.news[: min(limit, 10)]

        if not news:
            return f"No recent news found for '{resolved}'."

        headlines = []
        for i, article in enumerate(news, 1):
            content = article.get("content", {})
            title = content.get("title", "No title")
            provider = content.get("provider", {}).get("displayName", "Unknown source")
            headlines.append(f"{i}. [{provider}] {title}")

        return f"[LATEST NEWS — Yahoo Finance]\n" f"Ticker: {resolved}\n\n" + "\n".join(
            headlines
        )
    except Exception as e:
        return f"Error fetching news for '{ticker}': {str(e)}"

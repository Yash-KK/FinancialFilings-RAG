import yfinance as yf

from langchain.tools import tool
from yahoo_finance.schema import StockHistoryInput


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price and key financials for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, TSLA, MSFT
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return (
            f"Ticker: {ticker.upper()}\n"
            f"Company: {info.get('longName', 'N/A')}\n"
            f"Current Price: ${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}\n"
            f"Market Cap: ${info.get('marketCap', 'N/A'):,}\n"
            f"52W High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"52W Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"Dividend Yield: {info.get('dividendYield', 'N/A')}"
        )
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


@tool(args_schema=StockHistoryInput)
def get_stock_history(ticker: str, period: str = "1mo") -> str:
    """Get historical price performance of a stock over a given time period."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return f"No historical data found for {ticker}"

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        change_pct = ((end_price - start_price) / start_price) * 100

        return (
            f"Ticker: {ticker.upper()} | Period: {period}\n"
            f"Start Price: ${start_price:.2f}\n"
            f"End Price:   ${end_price:.2f}\n"
            f"Change:      {change_pct:+.2f}%\n"
            f"Period High: ${hist['High'].max():.2f}\n"
            f"Period Low:  ${hist['Low'].min():.2f}\n"
            f"Avg Volume:  {int(hist['Volume'].mean()):,}"
        )
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_stock_news(ticker: str) -> str:
    """Get the latest news headlines for a given stock ticker.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, GOOGL
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:2]
        if not news:
            return f"No news found for {ticker}"

        headlines = []
        for article in news:
            content = article.get("content", {})
            title = content.get("title", "No title")
            headlines.append(f"- {title}")

        return f"Latest news for {ticker.upper()}:\n" + "\n".join(headlines)
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"

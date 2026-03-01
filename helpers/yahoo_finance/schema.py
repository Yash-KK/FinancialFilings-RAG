from pydantic import BaseModel, Field
from typing import Literal


class StockHistoryInput(BaseModel):
    """Input for stock history queries."""

    ticker: str = Field(description="Stock ticker symbol e.g. AAPL, TSLA")
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"] = Field(
        default="1mo", description="Time period for historical data"
    )

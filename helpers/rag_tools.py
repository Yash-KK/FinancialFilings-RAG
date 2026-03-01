from helpers.common import vector_store, llm
from langchain.messages import HumanMessage, SystemMessage
from helpers.schema import ChunkMetadata
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain.tools import tool


def extract_filters(user_query: str):

    system_message = """
You extract metadata filters from financial queries.

Return None for fields not mentioned.

COMPANY MAPPINGS:
- Amazon/AMZN -> amazon
- Google/Alphabet/GOOGL/GOOG -> google
- Apple/AAPL -> apple
- Microsoft/MSFT -> microsoft
- Tesla/TSLA -> tesla
- Nvidia/NVDA -> nvidia
- Meta/Facebook/FB -> meta

DOC TYPE:
- Annual report -> 10-k
- Quarterly report -> 10-q
- Current report -> 8-k

Examples:
"Amazon Q3 2024 revenue" -> {"company_name": "amazon", "doc_type": "10-q", "fiscal_year": "2024", "fiscal_quarter": "q3"}
"Apple 2023 annual report" -> {"company_name": "apple", "doc_type": "10-k", "fiscal_year": "2023"}
"Tesla profitability" -> {"company_name": "tesla"}

Extract metadata strictly based on the user query.
"""

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_query),
    ]

    structured_llm = llm.with_structured_output(ChunkMetadata)
    try:
        response = structured_llm.invoke(messages)
        metadata = response.parsed if hasattr(response, "parsed") else response
        return metadata.model_dump(exclude_none=True)
    except Exception:
        return {}


@tool
def hybrid_search(query: str, k: int = 10):
    """
    Search historical financial documents (SEC filings: 10-K, 10-Q, 8-K) using hybrid search.

    **IMPORTANT: This is the PRIMARY tool for financial research.**
    **ALWAYS call this tool FIRST for ANY financial question unless:**
    - User explicitly asks for "current", "live", "real-time", or "latest" market data
    - User asks about current stock prices or today's market information

    This tool searches through:
    - Historical SEC filings (10-K annual reports, 10-Q quarterly reports)
    - Financial statements, revenue, expenses, cash flow data
    - Company performance metrics from past quarters and years
    - Automatically extracts filters (company, year, quarter, doc type) from your query

    Use this for queries about:
    - Historical revenue, profit, expenses ("What was Amazon's revenue in Q1 2024?")
    - Year-over-year or quarter-over-quarter comparisons
    - Financial metrics from SEC filings
    - Any historical financial data

    Args:
        query: Natural language search query (e.g., "Amazon Q1 2024 revenue")
        k: Number of results to return (default: 5)

    Returns:
        List of Document objects with page content and metadata (source_file, page_number, etc.)
    """

    filters = extract_filters(query)

    qdrant_filter = None

    if filters:
        condition = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]

        qdrant_filter = Filter(must=condition)

    results = vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)

    return results


# query = "what is amazon's cashflow in 2024 in q1?"
# results = hybrid_search(query, k=10)
# print("RESULTS:\n", results)

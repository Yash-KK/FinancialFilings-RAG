# Financial Documents RAG Pipeline

This project processes financial documents (e.g. 10-K, 10-Q, 8-K) from PDFs into a structured format and ingests them into a vector store for retrieval. The pipeline is split into four main notebooks plus a shared `helpers` module.

---

## Notebooks

### 1. `1_data_extraction.ipynb` — Data Extraction with Docling

Extracts content from PDFs into structured outputs suitable for later ingestion.

- **Input:** PDFs under `data/rag-data/pdfs` (e.g. `CompanyName DocType [Quarter] Year.pdf`).
- **Output:**
  - **Markdown:** Full document text with `<!-- page break -->` placeholders for page-based chunking.
  - **Images:** Pages with large charts/diagrams (>500×500 px) saved as images.
  - **Tables:** Each table is extracted with 2 paragraphs of context and page number, saved as markdown.

**Output layout:**

```
data/rag-data/markdown/{company}/{document}.md
data/rag-data/images/{company}/{document}/page_5.png
data/rag-data/tables/{company}/{document}/table_1_page_5.md
```

**Notable logic:**

- Uses **Docling** (via `helpers.doclingg`) to convert PDFs and extract tables.
- `extract_metadata_from_filename()` parses company, doc type, fiscal quarter/year from the filename.
- `extract_tables_with_context()` and `save_tables()` find tables in markdown and write them with context.
- `extract_pdf_content()` drives the flow: convert PDF → markdown, save tables, create dirs under `MARKDOWN_DIR` and `TABLES_DIR` from `helpers.common`.

Run this notebook first to populate `data/rag-data/markdown`, `data/rag-data/images`, and `data/rag-data/tables`.

---

### 2. `2_data_ingestion.ipynb` — Ingest into Qdrant

Takes the extracted markdown, tables, and image-description files and indexes them in **Qdrant** with hybrid (dense + sparse) embeddings.

- **Input:** Files under `data/rag-data/markdown`, `data/rag-data/tables`, and `data/rag-data/images_desc` (paths from `helpers.common`).
- **Vector store:** Qdrant at `http://localhost:6333`, collection `financial_docs_together`.
- **Embeddings:** Dense (HuggingFace `m2-bert`) + sparse (FastEmbed BM25) from `helpers.common`.

**Notable logic:**

- **Metadata from filenames:** Same idea as extraction — company, doc type, fiscal year/quarter from names like `CompanyName DocType [Quarter] Year`.
- **Chunking:** Markdown is split by `<!-- page break -->` into one chunk per page; tables and image descriptions are one chunk per file.
- **Deduplication:** SHA-256 file hashes are stored in payloads; `get_processed_hashes()` reads already-ingested hashes from the collection so `ingest_file_in_db()` can skip files already in the index.
- **Ingestion:** Each chunk is turned into a LangChain `Document` with metadata (company, doc type, year, quarter, page, content type, file hash, etc.) and added via `vector_store.add_documents()`.

Run this after extraction to build or update the Qdrant collection used for search.

---

### 3. `3_migration.ipynb` — Migrate Qdrant Collections

Copies documents from an existing Qdrant collection into a new collection (e.g. new name or re-index with the same embedding setup).

- **Input:** An existing collection (e.g. `pavestone_old`; name is set via `COLLECTION_NAME_`).
- **Output:** A new collection (e.g. `pavestone_old_v2`) with the same documents re-embedded using the same dense + sparse models from `helpers.common`.

**Notable logic:**

- **`extract_all_documents(collection_name)`:** Scrolls through the source collection (payloads + no vectors) and returns all points.
- **`points_to_documents(points)`:** Converts Qdrant points into LangChain `Document` objects (text from payload, metadata from payload fields).
- **`migrate_collection(old_name, new_name)`:** Fetches all points from `old_name`, converts to documents, creates a new Qdrant collection `new_name` with hybrid retrieval and `force_recreate=True`, and adds the documents so they get new vectors in the new collection.

Uses `client`, `vector_store`, `dense_embeddings`, `sparse_embeddings`, `URL`, and `COLLECTION_NAME_TOGETHER` from `helpers.common`, plus `QdrantVectorStore` and `RetrievalMode` for creating the target collection.

---

### 4. `4_data_retrieval.ipynb` — Query and Retrieve from Qdrant

Queries the indexed financial documents using natural language: extracts metadata filters from the user query, runs hybrid search with those filters, and optionally reranks results with a cross-encoder.

- **Input:** Natural-language questions (e.g. “What is Amazon’s revenue in 2023 Q1?”, “Tesla profitability”).
- **Vector store:** Same Qdrant collection as ingestion (`financial_docs_together` at `http://localhost:6333`).
- **Output:** Ranked list of LangChain `Document` chunks (optionally reranked).

**Notable logic:**

- **Metadata extraction:** `extract_filters(user_query)` uses the LLM (Together, via `helpers.common.llm`) with structured output (`ChunkMetadata` from `helpers.schema`) to infer filters from the query. It maps companies (e.g. Amazon/AMZN → `amazon`, Apple/AAPL → `apple`), doc types (e.g. “annual report” → `10-k`, “quarterly report” → `10-q`), and fiscal year/quarter. Returns a dict of non-`None` fields (e.g. `company_name`, `doc_type`, `fiscal_year`, `fiscal_quarter`) for use as Qdrant filters.
- **Hybrid search:** `hybrid_search(query, k=5)` calls `extract_filters(query)`, builds a Qdrant `Filter` from the result (`FieldCondition` + `MatchValue` per key under `metadata.*`), and runs `vector_store.similarity_search(query=..., k=..., filter=...)`. Returns a list of `Document` objects with metadata (company, doc type, year, quarter, page, content type, etc.).
- **Optional reranking:** `rerank_results(query, documents, top_k=10)` uses `HuggingFaceCrossEncoder` with `RERANKER_MODEL` (`BAAI/bge-reranker-base` from `helpers.common`) to score (query, doc) pairs and return the top `top_k` documents. Reranking is optional and can be applied after `hybrid_search` for better relevance.

Run this after ingestion when you want to search the indexed documents by natural language; ensure Qdrant is running and the collection is populated.

---

## Helpers (`helpers/`)

Shared configuration, schemas, and utilities used by the notebooks.

### `helpers/common.py`

- **Environment:** Loads `.env` and reads `TOGETHER_API_KEY`, `TOGETHER_BASE_URL`, `CHAT_MODEL`.
- **Paths:** `MARKDOWN_DIR`, `TABLES_DIR`, `IMAGES_DESC_DIR` for extracted data.
- **Qdrant:** `URL` (e.g. `http://localhost:6333`), `COLLECTION_NAME_TOGETHER` (`financial_docs_together`), `client` (`QdrantClient`), and a pre-built `vector_store` (`QdrantVectorStore` with hybrid retrieval).
- **Embeddings:** `dense_embeddings` (HuggingFace `togethercomputer/m2-bert-80M-8k-retrieval`) and `sparse_embeddings` (FastEmbed `Qdrant/bm25`).
- **Reranker:** `RERANKER_MODEL` — `BAAI/bge-reranker-base`, used by `4_data_retrieval.ipynb` for optional cross-encoder reranking.
- **LLM:** `llm` — `ChatOpenAI` wired to Together (base URL + API key from env).

Notebooks import from here instead of redefining clients, paths, or embedding models.

### `helpers/schema.py`

- **`DocType`:** Enum for document types (`10-k`, `10-q`, `8-k`).
- **`FiscalQuarter`:** Enum for quarters (`q1`–`q4`).
- **`ChunkMetadata`:** Pydantic model for chunk/document metadata: `company_name`, `doc_type`, `fiscal_year`, `fiscal_quarter` (with enum values).

Used where metadata is validated or passed in a structured form (e.g. filtering or typing).

### `helpers/doclingg.py`

- **`pdf_to_docling_converter`:** A Docling `DocumentConverter` configured for PDFs:
  - Table structure with TableFormer (FAST mode).
  - No OCR; optional page images and scaling.
  - CPU accelerator and local artifacts path.

Used by `1_data_extraction.ipynb` to convert PDFs to Docling documents and export markdown/tables.

---

## Setup

1. **Environment:** Create a `.env` with `TOGETHER_API_KEY`, `TOGETHER_BASE_URL`, and `CHAT_MODEL` (and ensure `.env` is in `.gitignore`).
2. **Qdrant:** Run Qdrant locally (e.g. Docker) so `http://localhost:6333` is available.
3. **Dependencies:** Install from `requirements.txt` (includes `docling`, `langchain-*`, `qdrant-client`, etc.) and use the same Python env when running the notebooks.

**Suggested order:** Run `1_data_extraction.ipynb` → `2_data_ingestion.ipynb`, then `4_data_retrieval.ipynb` to query the index. Use `3_migration.ipynb` only when you need to clone or re-index an existing Qdrant collection.

import os
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv


load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_BASE_URL = os.getenv("TOGETHER_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Paths
MARKDOWN_DIR = "data/rag-data/markdown"
TABLES_DIR = "data/rag-data/tables"
IMAGES_DESC_DIR = "data/rag-data/images_desc"

# Config
URL = "http://localhost:6333"
EMBEDDING_MODEL_TOGETHER = "togethercomputer/m2-bert-80M-8k-retrieval"
COLLECTION_NAME_TOGETHER = "financial_docs_together"

dense_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_TOGETHER,
    model_kwargs={"device": "cpu", "trust_remote_code": True},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 16,
    },
)


client = QdrantClient(url=URL)

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    url="http://localhost:6333",
    collection_name=COLLECTION_NAME_TOGETHER,
    retrieval_mode=RetrievalMode.HYBRID,
    force_recreate=False,
)

llm = ChatOpenAI(model=CHAT_MODEL, base_url=TOGETHER_BASE_URL, api_key=TOGETHER_API_KEY)

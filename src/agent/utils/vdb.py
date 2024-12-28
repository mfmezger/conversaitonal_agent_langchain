"""Vector Database Utilities for Qdrant integration."""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient, models
from ultra_simple_config import load_config

load_dotenv(override=True)

# Constants
SPARSE_MODEL_NAME = "Qdrant/bm25"
CONFIG_PATH = "config/main.yml"

# Initialize sparse embeddings
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)


def init_vdb(cfg: DictConfig, collection_name: str, embedding: Embeddings) -> QdrantVectorStore:
    """Initialize and return a QdrantVectorStore instance.

    Args:
    ----
        cfg (DictConfig): Configuration object containing Qdrant settings.
        collection_name (str): Name of the collection in Qdrant.
        embedding (Embeddings): Embedding model to use.

    Returns:
    -------
        QdrantVectorStore: Initialized vector store for the specified collection.

    """
    qdrant_client = _create_qdrant_client(cfg)
    logger.info(f"Using collection: {collection_name}")
    vector_db = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="fast-sparse-bm25",
    )
    logger.info("Qdrant DB initialized successfully.")
    return vector_db


@load_config(CONFIG_PATH)
def load_vec_db_conn(cfg: DictConfig) -> tuple[QdrantClient, DictConfig]:
    """Load the Vector Database Connection.

    Args:
    ----
        cfg (DictConfig): Configuration object with Qdrant settings.

    Returns:
    -------
        Tuple[QdrantClient, DictConfig]: QdrantClient instance and the original config.

    """
    return _create_qdrant_client(cfg)


def initialize_vector_db(collection_name: str, embeddings_size: int) -> None:
    """Initialize the vector database for a given collection.

    Args:
    ----
        collection_name (str): Name of the collection to initialize.
        embeddings_size (int): Size of the embeddings.

    """
    qdrant_client = load_vec_db_conn()

    if qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(f"Collection {collection_name} already exists.")
    else:
        _generate_collection(qdrant_client, collection_name, embeddings_size)


@load_config(CONFIG_PATH)
def initialize_all_vector_dbs(cfg: DictConfig) -> None:
    """Initialize all vector databases specified in the configuration."""
    collections = [
        (cfg.qdrant.collection_name_openai, cfg.openai_embeddings.size),
        (cfg.qdrant.collection_name_cohere, cfg.cohere_embeddings.size),
        (cfg.qdrant.collection_name_ollama, cfg.ollama_embeddings.size),
    ]

    for collection_name, embeddings_size in collections:
        initialize_vector_db(collection_name, embeddings_size)


def _create_qdrant_client(cfg: DictConfig) -> QdrantClient:
    """Create and return a QdrantClient instance.

    Args:
    ----
        cfg (DictConfig): Configuration object with Qdrant settings.

    Returns:
    -------
        QdrantClient: Initialized Qdrant client.

    """
    return QdrantClient(url=cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)


def _generate_collection(qdrant_client: QdrantClient, collection_name: str, embeddings_size: int) -> None:
    """Generate a new collection in Qdrant.

    Args:
    ----
        qdrant_client (QdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection to create.
        embeddings_size (int): Size of the embeddings.

    """
    qdrant_client.set_sparse_model(SPARSE_MODEL_NAME)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config=qdrant_client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"Collection {collection_name} created successfully.")

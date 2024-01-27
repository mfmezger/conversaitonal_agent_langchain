"""Vector Database Utilities."""
import os

from langchain_community.vectorstores import Qdrant
from langchain_core.embeddings import Embeddings
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from ultra_simple_config import load_config


def init_vdb(cfg: DictConfig, collection_name: str, embedding: Embeddings):
    """Establish a connection to the Qdrant DB.

    Args:
    ----
        cfg (DictConfig): the configuration from the file.
        collection_name (str): name of the collection in the Qdrant DB.
        embedding (Embeddings): Embedding Type.

    Returns:
    -------
        Qdrant: Established Connection to the Vector DB including Embeddings.
    """
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)

    logger.info(f"USING COLLECTION: {collection_name}")

    vector_db = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


@load_config("config/db.yml")
def load_vec_db_conn(cfg: DictConfig) -> QdrantClient:
    """Load the Vector Database Connection."""
    return QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc), cfg

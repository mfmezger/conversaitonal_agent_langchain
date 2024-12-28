"""Routes for the collection management."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.data_model.request_data_model import LLMProvider
from agent.utils.vdb import _generate_collection, load_vec_db_conn

router = APIRouter()


@load_config(location="config/main.yml")
def create_collection_in_vectordb(cfg: DictConfig, llm_provider: LLMProvider, collection_name: str) -> None:
    """Create a new collection in the Vector Database."""
    # get the embeddings size from the config file based on the LLM provider
    if llm_provider == LLMProvider.OPENAI:
        embeddings_size = cfg.openai_embeddings.size
    elif llm_provider == LLMProvider.COHERE:
        embeddings_size = cfg.cohere_embeddings.size
    elif llm_provider == LLMProvider.OLLAMA:
        embeddings_size = cfg.ollama_embeddings.size
    else:
        msg = f"Invalid LLM Provider: {llm_provider}"
        raise ValueError(msg)

    # load the vector database connection
    conn = load_vec_db_conn()

    _generate_collection(qdrant_client=conn, collection_name=collection_name, embeddings_size=embeddings_size)


@router.post("/create/{llm_provider}/{collection_name}", tags=["collection"])
def create_collection(llm_provider: LLMProvider, collection_name: str) -> JSONResponse:
    """Create a new collection.

    Args:
    ----
        llm_provider (LLMProvider): The LLM provider.
        collection_name (str): Name of the Qdrant Collection

    Returns:
    -------
        JSONResponse: Success Message.

    """
    create_collection_in_vectordb(llm_provider=llm_provider, collection_name=collection_name)
    return JSONResponse(content={"message": f"Collection {collection_name} created."})

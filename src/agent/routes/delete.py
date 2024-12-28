"""Route to handle the delection of a vector from the database."""

from fastapi import APIRouter
from loguru import logger
from qdrant_client import models
from qdrant_client.http.models.models import UpdateResult

from agent.data_model.request_data_model import LLMProvider
from agent.utils.vdb import load_vec_db_conn

router = APIRouter()


@router.delete("/delete/{llm_provider}/{page}/{source}", tags=["embeddings"])
def delete(page: int, source: str, llm_provider: LLMProvider = LLMProvider.OPENAI) -> UpdateResult:
    """Delete a vector from the database.

    Args:
    ----
        page (int): Number of the page in the document
        source (str): Name of the Document
        llm_provider (LLMProvider, optional): Which Large Language Model Provider. Defaults to LLMProvider.OPENAI.

    Raises:
    ------
        ValueError: Wrong LLM Provider

    Returns:
    -------
        UpdateResult: Result of the Update.

    """
    logger.info("Deleting Vector from Database")
    if llm_provider == LLMProvider.ALEPH_ALPHA:
        collection = "aleph-alpha"
    elif llm_provider == LLMProvider.OPENAI:
        collection = "openai"
    elif llm_provider == LLMProvider.GPT4ALL:
        collection = "gpt4all"
    else:
        msg = f"Unsupported LLM provider: {llm_provider}"
        raise ValueError(msg)
    qdrant_client = load_vec_db_conn()
    result = qdrant_client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.page", match=models.MatchValue(value=page)),
                    models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source)),
                ],
            )
        ),
    )
    logger.info("Deleted Point from Database via Metadata.")
    return result

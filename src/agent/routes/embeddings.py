"""Routes to manage embeddings."""

import asyncio
from pathlib import Path

from agent.backend.LLMStrategy import LLMContext, LLMStrategyFactory
from agent.data_model.request_data_model import EmbeddTextRequest, LLMBackend
from agent.data_model.response_data_model import EmbeddingResponse
from agent.utils.utility import create_tmp_folder
from fastapi import APIRouter, File, UploadFile
from loguru import logger

router = APIRouter()


@router.post("/documents", tags=["embeddings"])
async def post_embed_documents(llm_backend: LLMBackend, files: list[UploadFile] = File(...), file_ending: str = ".pdf") -> EmbeddingResponse:
    """Embeds multiple documents from files.

    Args:
    ----
        llm_backend (LLMBackend): Which LLM backend to use.
        files (list[UploadFile], optional): The uploaded files. Defaults to File(...).
        file_ending (str, optional): The file ending of the uploaded file. Defaults to ".pdf".

    Raises:
    ------
        ValueError: If a temporary folder to save the files is not provided.
        ValueError: If a file is not provided to save.

    Returns:
    -------
        EmbeddingResponse: Response containing the status and a list of file names.

    """
    logger.info("Embedding Multiple Documents")
    tmp_dir = create_tmp_folder()

    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, collection_name=llm_backend.collection_name))
    file_names = []

    # Use asyncio to write files concurrently
    write_tasks = []
    for file in files:
        file_name = file.filename
        if not file_name:
            msg = "Please provide a file to save."
            raise ValueError(msg)

        file_path = Path(tmp_dir) / file_name
        write_tasks.append(asyncio.to_thread(file.save, file_path))
        file_names.append(file_name)

    await asyncio.gather(*write_tasks)

    service.embed_documents(folder=tmp_dir, file_ending=file_ending)
    return EmbeddingResponse(status="success", files=file_names)


@router.post("/string/", tags=["embeddings"])
async def embedd_text(embedding: EmbeddTextRequest, llm_backend: LLMBackend) -> EmbeddingResponse:
    """Embedding text into the Vectordatabase.

    Args:
    ----
        embedding (EmbeddTextRequest): The Embedding Text with Metadata.
        llm_backend (LLMBackend): The LLM Backend to use.

    Returns:
    -------
        EmbeddingResponse: A response with Success or Failure Notification.

    """
    logger.info("Starting Embedding Text.")

    service = LLMContext(LLMStrategyFactory.get_strategy(strategy_type=llm_backend.llm_provider, collection_name=llm_backend.collection_name))

    tmp_dir = create_tmp_folder()
    with (Path(tmp_dir) / (embedding.file_name + ".txt")).open("w") as f:
        f.write(embedding.text)

    service.embed_documents(directory=tmp_dir, file_ending=".txt")
    return EmbeddingResponse(status="success", files=[embedding.file_name])

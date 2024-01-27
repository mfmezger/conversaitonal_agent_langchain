"""API Tests."""
import os
import shutil
from typing import TYPE_CHECKING

import httpx
import pytest
from fastapi.testclient import TestClient
from loguru import logger

from agent.api import app, create_tmp_folder

if TYPE_CHECKING:
    from httpx._models import Response

client: TestClient = TestClient(app)


def test_read_root() -> None:
    """Test the root method."""
    response: Response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to the Simple Aleph Alpha FastAPI Backend!"


def test_create_tmp_folder() -> None:
    """Test the create folder method."""
    tmp_dir = create_tmp_folder()
    assert Path(tmp_dir)
    shutil.rmtree(tmp_dir)


@pytest.mark.asyncio()
@pytest.mark.parametrize("provider", ["aa", "gpt4all"])  # TODO: if i get access again maybe also "openai",
async def test_upload_documents(provider: str) -> None:
    """Testing the upload of multiple documents."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        files = [
            open("tests/resources/1706.03762v5.pdf", "rb"),
            open("tests/resources/1912.01703v1.pdf", "rb"),
        ]
        if provider == "openai":
            logger.warning("Using OpenAI API")
            response: Response = await ac.post(
                "/embeddings/documents", params={"llm_backend": "openai", "token": os.getenv("OPENAI_API_KEY")}, files=[("files", file) for file in files]
            )
        elif provider == "aa":
            logger.warning("Using Aleph Alpha API")
            response: Response = await ac.post(
                "/embeddings/documents", params={"llm_backend": "aa", "token": os.getenv("ALEPH_ALPHA_API_KEY")}, files=[("files", file) for file in files]
            )
        elif provider == "gpt4all":
            response: Response = await ac.post(
                "/embeddings/documents", params={"llm_backend": "gpt4all", "token": os.getenv("ALEPH_ALPHA_API_KEY")}, files=[("files", file) for file in files]
            )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "files": ["1706.03762v5.pdf", "1912.01703v1.pdf"],
    }

    # Clean up temporary folders
    for entry in os.scandir():
        if entry.name.startswith("tmp_") and entry.is_dir():
            shutil.rmtree(entry.path)


@pytest.mark.asyncio()
@pytest.mark.parametrize("provider", ["aa", "gpt4all"])
async def test_embedd_one_document(provider: str) -> None:
    """Testing the upload of one document."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        tmp_file = open("tests/resources/1706.03762v5.pdf", "rb")

        if provider == "aa":
            logger.warning("Using Aleph Alpha API")
            response: Response = await ac.post(
                "/embeddings/documents",
                params={"llm_backend": "aa", "token": os.getenv("ALEPH_ALPHA_API_KEY")},
                files=[("files", tmp_file)],
            )
        elif provider == "gpt4all":
            response: Response = await ac.post(
                "/embeddings/documents",
                params={"llm_backend": "gpt4all", "token": os.getenv("ALEPH_ALPHA_API_KEY")},
                files=[("files", tmp_file)],
            )
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "files": ["1706.03762v5.pdf"],
    }

    # Clean up temporary folders
    for entry in os.scandir():
        if entry.name.startswith("tmp_") and entry.is_dir():
            shutil.rmtree(entry.path)


def test_search_route() -> None:
    """Testing with wrong backend."""
    response: Response = client.post(
        "/semantic/search",
        json={
            "query": "Was ist Vanilin",
            "llm_backend": {"llm_provider": "aa"},
            "filtering": {
                "threshold": 0,
                "collection_name": "aleph_alpha",
                "filter": {},
            },
            "collection_name": "string",
            "filter": {},
            "amount": 3,
            "threshold": 0,
        },
    )
    assert response.status_code == 200
    assert response.json() is not None


def test_embedd_text() -> None:
    """Test the embedd_text function."""
    # load text
    with open("tests/resources/file1.txt") as f:
        text = f.read()

    response: Response = client.post(
        "/embeddings/text/",
        json={"text": text, "llm_backend": {"llm_provider": "aa", "token": os.getenv("ALEPH_ALPHA_API_KEY")}, "file_name": "file", "seperator": "###"},
    )
    logger.info(response)
    assert response.status_code == 200
    logger.info(response.json())
    assert response.json() == {"message": "Text received and saved.", "filenames": "file"}

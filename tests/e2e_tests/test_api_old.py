"""API Tests."""
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.api import app

if TYPE_CHECKING:
    from httpx._models import Response
http_ok = 200
client: TestClient = TestClient(app)


def test_read_root() -> None:
    """Test the root method."""
    response: Response = client.get("/")
    assert response.status_code == http_ok


@pytest.mark.parametrize("provider", ["aa", "gpt4all", "openai", "cohere", "ollama"])
def test_create_collection(provider: str) -> None:
    """Test the create_collection function.

    Args:
    ----
        provider (str): The provider to use.
    """
    response: Response = client.post(
        f"/collection/create/{provider}/{uuid.uuid4()!s}",
    )
    assert response.status_code == http_ok


@pytest.mark.parametrize("provider", ["aa", "gpt4all", "openai"])
def test_semantic_search(provider: str) -> None:
    """Test the semantic_search function."""
    response: Response = client.post(
        "/semantic/search",
        json={
            "search": {"query": "What is Attention?", "amount": 3},
            "llm_backend": {"llm_provider": provider, "token": "", "collection_name": ""},
            "filtering": {"threshold": 0, "collection_name": "", "filter": {}},
        },
    )
    assert response.status_code == http_ok
    assert response.json() is not None


@pytest.mark.parametrize("provider", ["aa", "gpt4all", "openai"])
def test_embeddings_text(provider: str) -> None:
    """Test the embedd_text function."""
    # load text
    with Path("tests/resources/albert.txt").open() as f:
        text = f.read()

    response: Response = client.post(
        "/embeddings/string/",
        json={"embedding": {"text": text, "file_name": "file", "seperator": "###"}, "llm_backend": {"llm_provider": provider, "token": "", "collection_name": ""}},
    )
    assert response.status_code == http_ok


@pytest.mark.asyncio()
@pytest.mark.parametrize("provider", ["aa", "openai", "gpt4all"])
async def test_upload_documents(provider: str) -> None:
    """Testing the upload of multiple documents."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        with Path("tests/resources/1706.03762v5.pdf").open("rb") as file1, Path("tests/resources/1912.01703v1.pdf").open("rb") as file2:
            files = [file1, file2]
            response: Response = await ac.post(
                "/embeddings/documents",
                json={"llm_backend": {"llm_provider": provider, "token": "", "collection_name": ""}},
                params={"file_ending": ".pdf"},
                files=[("files", file) for file in files],
            )

    assert response.status_code == http_ok
    assert response.json() == {
        "status": "success",
        "files": ["1706.03762v5.pdf", "1912.01703v1.pdf"],
    }

    # Clean up temporary folders
    for entry in os.scandir():
        if entry.name.startswith("tmp_") and entry.is_dir():
            shutil.rmtree(entry.path)


@pytest.mark.asyncio()
@pytest.mark.parametrize("provider", ["aa", "openai", "gpt4all"])
async def test_embedd_one_document(provider: str) -> None:
    """Testing the upload of one document."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        with Path("tests/resources/1706.03762v5.pdf").open("rb") as tmp_file:
            # Use tmp_file here
            response: Response = await ac.post(
                "/embeddings/documents",
                json={"llm_backend": {"llm_provider": provider, "token": "", "collection_name": ""}},
                files=[("files", tmp_file)],
            )
        assert response.status_code == http_ok
        assert response.json() == {
            "status": "success",
            "files": ["1706.03762v5.pdf"],
        }

    # Clean up temporary folders
    for entry in os.scandir():
        if entry.name.startswith("tmp_") and entry.is_dir():
            shutil.rmtree(entry.path)

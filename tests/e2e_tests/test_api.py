"""API Tests."""
import os
import shutil

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx._models import Response
from loguru import logger

from agent.api import app, create_tmp_folder

client: TestClient = TestClient(app)


def test_read_root() -> None:
    """Test the root method."""
    response: Response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to the Simple Aleph Alpha FastAPI Backend!"


def test_create_tmp_folder() -> None:
    """Test the create folder method."""
    tmp_dir = create_tmp_folder()
    assert os.path.exists(tmp_dir)
    shutil.rmtree(tmp_dir)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["aleph-alpha", "gpt4all"])  # TODO: if i get access again maybe also "openai",
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
        elif provider == "aleph-alpha":
            logger.warning("Using Aleph Alpha API")
            response: Response = await ac.post(
                "/embeddings/documents", params={"llm_backend": "aleph-alpha", "token": os.getenv("ALEPH_ALPHA_API_KEY")}, files=[("files", file) for file in files]
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


@pytest.mark.asyncio
async def test_embedd_one_document() -> None:
    """Testing the upload of one document."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        tmp_file = open("tests/resources/1706.03762v5.pdf", "rb")
        response: Response = await ac.post(
            "/embeddings/document/", params={"llm_backend": "aleph-alpha", "token": os.getenv("ALEPH_ALPHA_API_KEY")}, files=[("file", tmp_file)]
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


def test_search_route_invalid_provider() -> None:
    """Testing with wrong backend."""
    with pytest.raises(ValueError):
        response: Response = client.post(
            "/semantic/search",
            json={
                "query": "example query",
                "llm_backend": "invalid_provider",
                "token": "example_token",
                "amount": 3,
            },
        )
        assert response.status_code == 400
        assert "ValueError" in response.text


def test_search_route() -> None:
    """Testing with wrong backend."""
    response: Response = client.post(
        "/semantic/search",
        json={
            "query": "Was ist Vanilin?",
            "llm_backend": "aa",
            "amount": 3,
        },
    )
    assert response.status_code == 200
    assert response.json() is not None


def test_explain_output() -> None:
    """Test the function with valid arguments."""
    response: Response = client.post(
        "/explaination/aleph_alpha_explain", json={"prompt": "What is the capital of France?", "output": "Paris", "token": os.getenv("ALEPH_ALPHA_API_KEY")}
    )
    assert response.status_code == 200


def test_wrong_input_explain_output() -> None:
    """Test the function with wrong arguments."""
    with pytest.raises(ValueError):
        client.post("/explaination/aleph_alpha_explain", json={"prompt": "", "output": "", "token": os.getenv("ALEPH_ALPHA_API_KEY")})
    with pytest.raises(ValueError):
        client.post("/explaination/aleph_alpha_explain", json={"prompt": "", "output": "asdfasdf", "token": os.getenv("ALEPH_ALPHA_API_KEY")})
    with pytest.raises(ValueError):
        client.post("/explaination/aleph_alpha_explain", json={"prompt": "asdfasdf", "output": "", "token": os.getenv("ALEPH_ALPHA_API_KEY")})


def test_embedd_text() -> None:
    """Test the embedd_text function."""
    # load text
    with open("tests/resources/file1.txt") as f:
        text = f.read()

    response: Response = client.post(
        "/embeddings/text/", json={"text": text, "llm_backend": "aa", "file_name": "file", "token": os.getenv("ALEPH_ALPHA_API_KEY"), "seperator": "###"}
    )
    logger.info(response)
    assert response.status_code == 200
    logger.info(response.json())
    assert response.json() == {"message": "Text received and saved.", "filenames": "file"}

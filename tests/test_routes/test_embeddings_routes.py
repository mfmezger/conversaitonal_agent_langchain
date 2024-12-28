import pytest
import os
import requests
from agent.utils.vdb import load_vec_db_conn
from fastapi.testclient import TestClient
from agent.api import app

client = TestClient(app)

# @pytest.mark.parametrize("llm_provider", ["openai", "cohere", "ollama"])
# def test_create_collection(llm_provider: str):
#     response = client.post(f"/embeddings/documents", json={"llm_backend": {"llm_provider": llm_provider, "collection_name": "test"}})

#     assert response.status_code == 200
#     assert response.json() == {"message": "Collection test created."}









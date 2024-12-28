import pytest
from fastapi.testclient import TestClient
from agent.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

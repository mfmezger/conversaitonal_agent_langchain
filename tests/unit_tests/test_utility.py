"""Test cases for utility module."""

import os
from pathlib import Path
import pytest
from langchain_core.documents import Document

from agent.utils.utility import (
    combine_text_from_list,
    load_prompt_template,
    convert_qdrant_result_to_retrieval_results,
    create_tmp_folder,
    format_docs_for_citations,
    check_env_variables,
    MissingEnvironmentVariableError,
)
from agent.data_model.internal_model import RetrievalResults


def test_combine_text_from_list_success():
    """Test successful combination of text list."""
    input_list = ["Hello", "World", "Test"]
    result = combine_text_from_list(input_list)
    assert result == "Hello\nWorld\nTest"


def test_combine_text_from_list_empty():
    """Test combining empty list."""
    result = combine_text_from_list([])
    assert result == ""


def test_combine_text_from_list_single():
    """Test combining single item list."""
    result = combine_text_from_list(["Hello"])
    assert result == "Hello"


def test_combine_text_from_list_invalid_input():
    """Test error handling for invalid input."""
    with pytest.raises(TypeError, match="Input list must contain only strings"):
        combine_text_from_list(["Hello", 123, "World"])


def test_load_prompt_template_not_found():
    """Test error handling for non-existent prompt file."""
    with pytest.raises(FileNotFoundError):
        load_prompt_template("nonexistent.txt", "test_task")


def test_convert_qdrant_result_to_retrieval_results():
    """Test conversion of Qdrant results to RetrievalResults."""
    # Create test data
    doc1 = Document(page_content="Test content 1", metadata={"source": "test1"})
    doc2 = Document(page_content="Test content 2", metadata={"source": "test2"})
    qdrant_results = [(doc1, 0.8), (doc2, 0.9)]

    # Convert and verify
    results = convert_qdrant_result_to_retrieval_results(qdrant_results)

    assert len(results) == 2
    assert isinstance(results[0], RetrievalResults)
    assert results[0].document == "Test content 1"
    assert results[0].score == 0.8
    assert results[0].metadata == {"source": "test1"}
    assert results[1].document == "Test content 2"
    assert results[1].score == 0.9
    assert results[1].metadata == {"source": "test2"}


def test_create_tmp_folder(tmp_path):
    """Test temporary folder creation."""
    # Temporarily change working directory to tmp_path
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    try:
        tmp_dir = create_tmp_folder()
        assert tmp_dir.exists()
        assert tmp_dir.is_dir()
        assert str(tmp_dir).startswith(str(tmp_path))
        assert "tmp_" in str(tmp_dir)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def test_format_docs_for_citations():
    """Test formatting documents for citations."""
    docs = [
        Document(page_content="Test content 1"),
        Document(page_content="Test content 2")
    ]
    result = format_docs_for_citations(docs)
    expected = "<doc id='0'>Test content 1</doc>\n<doc id='1'>Test content 2</doc>"
    assert result == expected


def test_format_docs_for_citations_empty():
    """Test formatting empty document list."""
    result = format_docs_for_citations([])
    assert result == ""


def test_check_env_variables_success(monkeypatch):
    """Test successful environment variable check."""
    monkeypatch.setenv("TEST_VAR1", "value1")
    monkeypatch.setenv("TEST_VAR2", "value2")

    # Should not raise any exception
    check_env_variables(["TEST_VAR1", "TEST_VAR2"])


def test_check_env_variables_missing(monkeypatch):
    """Test error handling for missing environment variables."""
    monkeypatch.setenv("TEST_VAR1", "value1")
    # TEST_VAR2 is not set

    with pytest.raises(MissingEnvironmentVariableError) as exc_info:
        check_env_variables(["TEST_VAR1", "TEST_VAR2"])

    assert "TEST_VAR2" in str(exc_info.value)


def test_missing_environment_variable_error():
    """Test MissingEnvironmentVariableError creation."""
    missing_vars = ["VAR1", "VAR2"]
    error = MissingEnvironmentVariableError(missing_vars)

    assert error.missing_vars == missing_vars
    assert "VAR1" in str(error)
    assert "VAR2" in str(error)

"""Test module for prompts.py functionality."""

import os
import pytest
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock
from omegaconf import DictConfig

from agent.utils.prompts import load_prompt_from_file, load_prompts


@pytest.fixture
def sample_prompt_content():
    return "This is a test prompt template\nWith multiple lines\n{variable}"


@pytest.fixture
def mock_config():
    return DictConfig({
        "prompts": {
            "response_template": "prompts/response_template.txt",
            "cohere_response_template": "prompts/cohere_response_template.txt",
            "rephrase_template": "prompts/rephrase_template.txt"
        }
    })


def test_load_prompt_from_file_success(tmp_path, sample_prompt_content):
    """Test successful loading of a prompt file."""
    # Create a temporary file with test content
    test_file = tmp_path / "test_prompt.txt"
    test_file.write_text(sample_prompt_content)

    # Load and verify content
    result = load_prompt_from_file(str(test_file))
    assert result == sample_prompt_content


def test_load_prompt_from_file_not_found():
    """Test handling of non-existent prompt file."""
    with pytest.raises(FileNotFoundError):
        load_prompt_from_file("nonexistent_file.txt")


def test_load_prompt_from_file_permission_error(tmp_path):
    """Test handling of permission error when reading prompt file."""
    test_file = tmp_path / "no_permission.txt"
    test_file.write_text("test content")

    # Remove read permissions
    os.chmod(test_file, 0o000)

    with pytest.raises(IOError):
        load_prompt_from_file(str(test_file))

    # Restore permissions for cleanup
    os.chmod(test_file, 0o666)


def test_load_prompts_invalid_config():
    """Test handling of invalid configuration."""
    invalid_config = DictConfig({"prompts": {}})

    with pytest.raises(TypeError):
        load_prompts(invalid_config)

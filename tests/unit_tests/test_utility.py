"""Tests for the utility functions."""
import pytest

from agent.utils.utility import generate_prompt, validate_token


def test_generate_prompt() -> None:
    """Test that generate_prompt returns the correct prompt."""
    text = "blubby"
    query = "blubby2"

    prompt = generate_prompt(prompt_name="qa.j2", text=text, query=query, language="de")
    print(f"Prompt: {prompt}")
    # assert prompt contains text and query
    assert text in prompt
    assert query in prompt


def test_combine_text_from_list() -> None:
    """Test that combine_text_from_list returns the correct text."""
    pass


def test_validate_token() -> None:
    """Test that validate_token returns the correct token."""
    token = validate_token(token="example_token", llm_backend="openai", aleph_alpha_key="example_key_a", openai_key="example_key_o")

    assert token == "example_token"

    token = validate_token(token="", llm_backend="aleph-alpha", aleph_alpha_key="example_key_a", openai_key="example_key_o")

    assert token == "example_key_a"

    token = validate_token(token="", llm_backend="openai", aleph_alpha_key="example_key_a", openai_key="example_key_o")

    assert token == "example_key_o"

    token = validate_token(token=None, llm_backend="openai", aleph_alpha_key="example_key_a", openai_key="example_key_o")

    assert token == "example_key_o"

    token = validate_token(token="", llm_backend="gpt4all", aleph_alpha_key="example_key_a", openai_key="example_key_o")

    assert token == "gpt4all"

    with pytest.raises(ValueError):
        token = validate_token(token=None, llm_backend="openai", aleph_alpha_key=None, openai_key=None)

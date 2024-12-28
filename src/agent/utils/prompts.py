"""Prompt Management Module.

This module provides functionality for loading and managing prompt templates used in the application.
It handles loading prompts from files and configuring them based on YAML configuration.
"""

from pathlib import Path

from omegaconf import DictConfig
from ultra_simple_config import load_config


def load_prompt_from_file(path: str) -> str:
    """Load a prompt template from a file.

    Args:
    ----
        path (Path): Path to the prompt template file.

    Returns:
    -------
        str: The content of the prompt template file.

    Raises:
    ------
        FileNotFoundError: If the specified file does not exist.
        IOError: If there are issues reading the file.

    """
    with Path(path).open("r") as file:
        return file.read()


@load_config("config/prompts.yaml")
def load_prompts(cfg: DictConfig) -> tuple[str, str, str]:
    """Load all prompt templates based on configuration.

    This function loads three different types of prompt templates:
    - Response template: Used for standard responses
    - Cohere response template: Specific template for Cohere API responses
    - Rephrase template: Used for rephrasing queries or responses

    Args:
    ----
        cfg (DictConfig): Configuration object containing prompt file paths
                         under cfg.prompts.{response_template, cohere_response_template, rephrase_template}

    Returns:
    -------
        tuple[str, str, str]: A tuple containing:
            - response_template (str): The standard response template
            - cohere_response_template (str): Template for Cohere responses
            - rephrase_template (str): Template for rephrasing

    Raises:
    ------
        FileNotFoundError: If any of the template files are missing
        IOError: If there are issues reading any of the template files

    """
    response_template = load_prompt_from_file(cfg.prompts.response_template)
    cohere_response_template = load_prompt_from_file(cfg.prompts.cohere_response_template)
    rephrase_template = load_prompt_from_file(cfg.prompts.rephrase_template)

    return response_template, cohere_response_template, rephrase_template

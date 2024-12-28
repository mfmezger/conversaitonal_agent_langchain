"""Utility module."""

import os
import uuid
from collections.abc import Sequence
from pathlib import Path

from agent.data_model.internal_model import RetrievalResults
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from lingua import Language, LanguageDetectorBuilder
from loguru import logger

# Constants
LANGUAGES = [Language.ENGLISH, Language.GERMAN]
DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).with_minimum_relative_distance(0.7).build()


class MissingEnvironmentVariableError(Exception):
    """Custom error for missing environment variables."""

    def __init__(self, missing_vars: list[str]) -> None:
        """Init the custom Error."""
        self.missing_vars = missing_vars
        self.message = f"Missing required environment variables: {', '.join(missing_vars)}"
        super().__init__(self.message)


def combine_text_from_list(input_list: list[str]) -> str:
    """Combines all strings in a list to one string."""
    if not all(isinstance(item, str) for item in input_list):
        msg = "Input list must contain only strings"
        raise TypeError(msg)

    logger.info(f"List: {input_list}")
    return "\n".join(input_list)


def load_prompt_template(prompt_name: str, task: str) -> PromptTemplate:
    """Loading a task specific prompt template."""
    prompt_path = Path("prompts") / task / prompt_name
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        msg = f"Prompt file '{prompt_name}' not found."
        raise FileNotFoundError(msg) from e


def convert_qdrant_result_to_retrieval_results(docs: list[tuple]) -> list[RetrievalResults]:
    """Converts the Qdrant result to a list of RetrievalResults."""
    return [RetrievalResults(document=doc[0].page_content, score=doc[1], metadata=doc[0].metadata) for doc in docs]


def create_tmp_folder() -> Path:
    """Creates a temporary folder for files to store."""
    tmp_dir = Path.cwd() / f"tmp_{uuid.uuid4()}"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new folder {tmp_dir}.")
        return tmp_dir
    except Exception as e:
        logger.error(f"Failed to create directory {tmp_dir}. Error: {e}")
        raise


def format_docs_for_citations(docs: Sequence[Document]) -> str:
    """Format the documents for citations."""
    return "\n".join(f"<doc id='{i}'>{doc.page_content}</doc>" for i, doc in enumerate(docs))


def check_env_variables(required_vars: list[str]) -> None | MissingEnvironmentVariableError:
    """Check if all required environment variables are set."""
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise MissingEnvironmentVariableError(missing_vars)

"""Utils for Ingestion."""
import os

from aleph_alpha_client import Client
from dotenv import load_dotenv

load_dotenv()

aleph_alpha_token = os.getenv("ALEPH_ALPHA_API_KEY")

client = Client(token=aleph_alpha_token)
tokenizer = client.tokenizer("luminous-base")


def split_text(text: str) -> list:
    """Split the text into chunks.

    Args:
    ----
        text (str): input text.

    Returns:
    -------
        List: List of splits.
    """
    # define the metadata for the document
    return splitter.split_text(text)


def count_tokens(text: str) -> int:
    """Count the number of tokens in the text.

    Args:
    ----
        text (str): The text to count the tokens for.

    Returns:
    -------
        int: Number of tokens.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)

"""LLMBase Module: Defines the abstract base class for Language Model implementations.

This module implements the Strategy Pattern for different LLM backends.
"""

from abc import ABC, abstractmethod

from agent.data_model.request_data_model import LLMBackend, SearchParams


class LLMBase(ABC):
    """Abstract base class for Language Model implementations.

    This class defines the interface for various LLM strategies, following the Strategy Pattern.
    Concrete implementations should inherit from this class and implement all abstract methods.

    Attributes
    ----------
        collection_name (Optional[str]): The name of the collection in the vector database.

    """

    @abstractmethod
    def __init__(self, collection_name: str | None) -> None:
        """Initialize the LLM Base.

        Args:
        ----
            collection_name (Optional[str]): The name of the collection to be used.

        """
        self.collection_name: str | None = collection_name

    @abstractmethod
    def embed_documents(self, directory: str, llm_backend: LLMBackend) -> None:
        """Embed new documents in the vector database.

        Args:
        ----
            directory (str): The directory containing the documents to be embedded.
            llm_backend (LLMBackend): The LLM backend to use for embedding.

        Raises:
        ------
            NotImplementedError: This method should be implemented by subclasses.

        """
        msg = "Subclasses must implement embed_documents method"
        raise NotImplementedError(msg)

    @abstractmethod
    def search(self, search: SearchParams) -> list[dict]:
        """Search the documents in the vector database using semantic search.

        Args:
        ----
            search (SearchParams): The search parameters.

        Returns:
        -------
            List[dict]: A list of search results, where each result is a dictionary.

        Raises:
        ------
            NotImplementedError: This method should be implemented by subclasses.

        """
        msg = "Subclasses must implement search method"
        raise NotImplementedError(msg)

    @abstractmethod
    def summarize_text(self, text: str) -> str:
        """Summarize the given text.

        Args:
        ----
            text (str): The text to be summarized.

        Returns:
        -------
            str: The summarized text.

        Raises:
        ------
            NotImplementedError: This method should be implemented by subclasses.

        """
        msg = "Subclasses must implement summarize_text method"
        raise NotImplementedError(msg)

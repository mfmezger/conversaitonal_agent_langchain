"""The Strategy Pattern for the LLM Backend."""

from typing import ClassVar

from agent.backend.LLMBase import LLMBase
from agent.backend.services.cohere_service import CohereService
from agent.backend.services.ollama_service import OllamaService
from agent.backend.services.open_ai_service import OpenAIService
from agent.data_model.request_data_model import LLMProvider, SearchParams


class LLMStrategyFactory:
    """The Factory to select the correct LLM Backend.

    Raises
    ------
        ValueError: If Provider is not known.

    Returns
    -------
        Strategy: The correct strategy.

    """

    _strategies: ClassVar = {
        LLMProvider.OPENAI: OpenAIService,
        LLMProvider.COHERE: CohereService,
        LLMProvider.OLLAMA: OllamaService,
    }

    @staticmethod
    def get_strategy(strategy_type: str, collection_name: str) -> LLMBase:
        """Get the correct strategy.

        Args:
        ----
            strategy_type (str): The strategy type.
            token (str): The token for the strategy.
            collection_name (str): The collection name of the vector database.

        Raises:
        ------
            ValueError: If there is an unkown provider.

        Returns:
        -------
            LLMBase: Selected Strategy.

        """
        strategy = LLMStrategyFactory._strategies.get(strategy_type)
        if strategy is None:
            msg = "Unknown Strategy Type"
            raise ValueError(msg)
        return strategy(collection_name=collection_name)


class LLMContext:
    """The Context for the LLM Backend."""

    def __init__(self, llm: LLMBase) -> None:
        """Init the Context."""
        self.llm = llm

    def change_strategy(self, strategy_type: str, collection_name: str) -> None:
        """Changes the strategy using the Factory."""
        self.llm = LLMStrategyFactory.get_strategy(strategy_type=strategy_type, collection_name=collection_name)

    def search(self, search: SearchParams) -> list:
        """Wrapper for the search."""
        return self.llm.search(search=search)

    def embed_documents(self, directory: str, file_ending: str) -> None:
        """Wrapper for the Embedding of Documents."""
        return self.llm.embed_documents(directory=directory, file_ending=file_ending)

    def summarize_text(self, text: str) -> str:
        """Wrapper for the summarization of text."""
        return self.llm.summarize_text(text)

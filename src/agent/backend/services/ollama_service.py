"""Ollama Backend Service for document embedding and retrieval."""

import os

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import SearchParams
from agent.utils.vdb import init_vdb
from dotenv import load_dotenv
from langchain.document_loaders.base import BaseLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import chain
from langchain_text_splitters import NLTKTextSplitter, TextSplitter
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

load_dotenv()


class OllamaService(LLMBase):
    """Wrapper for Ollama language models, providing document embedding and retrieval services."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str | None = None) -> None:
        """Initialize the Ollama Service.

        Args:
        ----
            cfg (DictConfig): Configuration object.
            collection_name (Optional[str]): Name of the vector database collection.

        """
        super().__init__(collection_name=collection_name)
        self.cfg = cfg
        self.collection_name = collection_name or self.cfg.qdrant.collection_name_ollama
        self.embedding = OllamaEmbeddings(model=self.cfg.ollama_embeddings.embedding_model_name)
        self.vector_db = init_vdb(collection_name=self.collection_name, embedding=self.embedding)

    def embed_documents(self, directory: str, file_ending: str = ".pdf") -> None:
        """Embed documents from the given directory into the vector database.

        Args:
        ----
            directory (str): Path to the directory containing documents.
            file_ending (str): File extension of the documents to be processed.

        Raises:
        ------
            ValueError: If an unsupported file ending is provided.

        """
        loader = self._create_directory_loader(directory, file_ending)
        splitter = self._create_text_splitter()
        docs = self._load_and_split_documents(loader, splitter)
        self._add_documents_to_vector_db(docs)

    def _create_directory_loader(self, directory: str, file_ending: str) -> DirectoryLoader:
        """Create a directory loader with the specified file ending."""
        loader_cls = self._get_loader_class(file_ending)
        return DirectoryLoader(directory, glob=f"*{file_ending}", loader_cls=loader_cls)

    def _create_text_splitter(self) -> TextSplitter:
        """Create a text splitter with the specified parameters."""
        return NLTKTextSplitter(length_function=len, chunk_size=500, chunk_overlap=75)

    def _load_and_split_documents(self, loader: DirectoryLoader, splitter: TextSplitter) -> list[Document]:
        """Load and split documents using the specified loader and splitter."""
        docs = loader.load_and_split(splitter)
        logger.info(f"Loaded {len(docs)} documents.")
        return docs

    def _add_documents_to_vector_db(self, docs: list[Document]) -> None:
        """Add the loaded documents to the vector database."""
        text_list = [doc.page_content for doc in docs]
        metadata_list = [self._process_metadata(doc.metadata) for doc in docs]
        self.vector_db.add_texts(texts=text_list, metadatas=metadata_list)
        logger.info("SUCCESS: Texts embedded.")

    def create_search_chain(self, search: SearchParams) -> BaseRetriever:
        """Create a search chain for semantic search in the Qdrant DB.

        Args:
        ----
            search (SearchParams): Search parameters.

        Returns:
        -------
            BaseRetriever: A retriever chain for semantic search.

        """

        @chain
        def retriever_with_score(query: str) -> list[Document]:
            """Define a retriever that returns documents with their similarity scores.

            Args:
            ----
                query (str): User's search query.

            Returns:
            -------
                List[Document]: List of relevant documents with similarity scores.

            """
            docs_and_scores = self.vector_db.similarity_search_with_score(query, k=search.k, filter=search.filter, score_threshold=search.score_threshold)
            for doc, score in docs_and_scores:
                doc.metadata["score"] = score
            return [doc for doc, _ in docs_and_scores]

        return retriever_with_score

    def summarize_text(self, text: str) -> str:
        """Summarize the given text.

        Args:
        ----
            text (str): Text to be summarized.

        Returns:
        -------
            str: Summarized text.

        Note:
        ----
            This method is not implemented yet.

        """
        # TODO: Implement text summarization
        msg = f"Text summarization is not implemented yet. {text}"
        raise NotImplementedError(msg)

    @staticmethod
    def _get_loader_class(file_ending: str) -> type[BaseLoader]:
        """Get the appropriate document loader class based on file extension.

        Args:
        ----
            file_ending (str): File extension.

        Returns:
        -------
            Type[BaseLoader]: Loader class for the specified file type.

        Raises:
        ------
            ValueError: If an unsupported file ending is provided.

        """
        loaders = {".pdf": PyPDFium2Loader, ".txt": TextLoader}
        if file_ending not in loaders:
            msg = f"Unsupported file ending: {file_ending}"
            raise ValueError(msg)
        return loaders[file_ending]

    @staticmethod
    def _process_metadata(metadata: dict) -> dict:
        """Process document metadata, simplifying the source path.

        Args:
        ----
            metadata (dict): Original metadata.

        Returns:
        -------
            dict: Processed metadata.

        """
        if "source" in metadata and "/" in metadata["source"]:
            metadata["source"] = os.path.basename(metadata["source"])
        return metadata


if __name__ == "__main__":
    # Example usage
    query = "Was ist Attention?"
    ollama_service = OllamaService(collection_name="")
    ollama_service.embed_documents(directory="tests/resources/")
    chain = ollama_service.create_search_chain(SearchParams())
    answer = chain.invoke(query)
    logger.info(answer)

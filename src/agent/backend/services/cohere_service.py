"""Cohere Backend Service for document embedding and retrieval."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import chain
from langchain_text_splitters import NLTKTextSplitter
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import SearchParams
from agent.utils.vdb import init_vdb

load_dotenv(override=True)


class CohereService(LLMBase):
    """Wrapper for Cohere LLMs, providing document embedding and retrieval services."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str | None = None) -> None:
        """Initialize the Cohere Service.

        Args:
        ----
            cfg (DictConfig): Configuration object.
            collection_name (Optional[str]): Name of the vector database collection.
                If not provided, uses the default from the configuration.

        """
        super().__init__(collection_name=collection_name)

        self.cfg = cfg
        self.collection_name = collection_name or self.cfg.qdrant.collection_name_cohere

        self.embedding = CohereEmbeddings(model=self.cfg.cohere_embeddings.embedding_model_name)
        self.vec_db = init_vdb(collection_name=self.collection_name, embedding=self.embedding)

    def embed_documents(self, directory: str, file_ending: str = ".pdf") -> None:
        """Embed documents from the specified directory into the vector database.

        Args:
        ----
            directory (str): Path to the directory containing documents.
            file_ending (str): File extension of the documents to process. Defaults to ".pdf".

        Raises:
        ------
            ValueError: If an unsupported file ending is provided.

        """
        loader = self._get_document_loader(directory, file_ending)
        splitter = NLTKTextSplitter(length_function=len, chunk_size=500, chunk_overlap=75)

        docs = loader.load_and_split(splitter)
        logger.info(f"Loaded {len(docs)} documents.")

        text_list, metadata_list = self._prepare_documents(docs)
        self.vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Texts embedded.")

    def create_search_chain(self, search: SearchParams) -> BaseRetriever:
        """Create a search chain for document retrieval.

        Args:
        ----
            search (SearchParams): Search parameters.

        Returns:
        -------
            BaseRetriever: A retriever chain that can be invoked for document search.

        """

        @chain
        def retriever_with_score(query: str) -> list[Document]:
            """Define a retriever that returns documents with their similarity scores.

            Args:
            ----
                query (str): The search query.

            Returns:
            -------
                List[Document]: List of retrieved documents with scores in metadata.

            """
            docs, scores = zip(
                *self.vector_db.similarity_search_with_score(query, k=search.k, filter=search.filter, score_threshold=search.score_threshold), strict=False
            )
            for doc, score in zip(docs, scores, strict=False):
                doc.metadata["score"] = score

            return list(docs)

        return retriever_with_score

    def summarize_text(self, text: str) -> str:
        """Summarize the given text.

        Args:
        ----
            text (str): The text to summarize.

        Returns:
        -------
            str: The summarized text.

        Note:
        ----
            This method is not implemented yet.

        """
        # TODO: Implement text summarization

        msg = "Text summarization is not implemented yet."
        raise NotImplementedError(msg)

    def _get_document_loader(self, directory: str, file_ending: str) -> DirectoryLoader:
        """Get the appropriate document loader based on file ending.

        Args:
        ----
            directory (str): Directory path.
            file_ending (str): File extension.

        Returns:
        -------
            DirectoryLoader: Configured document loader.

        Raises:
        ------
            ValueError: If an unsupported file ending is provided.

        """
        if file_ending == ".pdf":
            return DirectoryLoader(directory, glob=f"*{file_ending}", loader_cls=PyPDFium2Loader)
        elif file_ending == ".txt":
            return DirectoryLoader(directory, glob=f"*{file_ending}", loader_cls=TextLoader)
        else:
            msg = f"Unsupported file ending: {file_ending}"
            raise ValueError(msg)

    def _prepare_documents(self, docs: list[Document]) -> tuple[list[str], list[dict]]:
        """Prepare documents for embedding by extracting text and metadata.

        Args:
        ----
            docs (List[Document]): List of loaded documents.

        Returns:
        -------
            Tuple[List[str], List[dict]]: Tuple containing lists of text content and metadata.

        """
        text_list = [doc.page_content for doc in docs]
        metadata_list = [doc.metadata for doc in docs]

        for metadata in metadata_list:
            if "/" in metadata["source"]:
                metadata["source"] = Path(metadata["source"]).name

        return text_list, metadata_list


if __name__ == "__main__":
    # Example usage
    query = "Was ist Attention?"
    cohere_service = CohereService(collection_name="")
    cohere_service.embed_documents(directory="tests/resources/")
    chain = cohere_service.create_search_chain(SearchParams(query=query, amount=3))
    answer = chain.invoke(query)
    logger.info(answer)

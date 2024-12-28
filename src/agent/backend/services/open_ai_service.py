"""OpenAI service for initializing and interacting with Qdrant vector database."""

import os
from typing import Any

import openai
from dotenv import load_dotenv
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.backend.LLMBase import LLMBase
from agent.data_model.request_data_model import SearchParams
from agent.utils.utility import load_prompt_template
from agent.utils.vdb import init_vdb

load_dotenv()


class OpenAIService(LLMBase):
    """OpenAI Backend Service for vector database operations and text processing."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str = "") -> None:
        """Initialize the OpenAI Service.

        Args:
        ----
            cfg (DictConfig): Configuration object.
            collection_name (str, optional): Name of the vector database collection. Defaults to "".

        """
        super().__init__(collection_name=collection_name)

        self.cfg = cfg
        self.collection_name = collection_name or self.cfg.qdrant.collection_name_openai

        # self._initialize_prompt()
        self._initialize_embedding()
        self._initialize_vector_db()

        # initialize the search chain.

    def _initialize_prompt(self) -> None:
        """Initialize the chat prompt template."""
        template = load_prompt_template(prompt_name="cohere_chat.j2", task="chat")
        self.prompt = ChatPromptTemplate.from_template(template=template, template_format="jinja2")

    def _initialize_embedding(self) -> None:
        """Initialize the embedding model based on configuration."""
        if self.cfg.openai_embeddings.azure:
            self.embedding = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment=self.cfg.openai_embeddings.embedding_model_name,
                openai_api_version=self.cfg.openai_embeddings.openai_api_version,
            )
        else:
            self.embedding = OpenAIEmbeddings(model=self.cfg.openai_embeddings.embedding_model_name)

    def _initialize_vector_db(self) -> None:
        """Initialize the vector database."""
        self.vector_db = init_vdb(collection_name=self.collection_name, embedding=self.embedding)

    def embed_documents(self, directory: str, file_ending: str = ".pdf") -> None:
        """Embed documents from the given directory into the vector database.

        Args:
        ----
            directory (str): Directory containing the documents.
            file_ending (str, optional): File extension of the documents. Defaults to ".pdf".

        Raises:
        ------
            ValueError: If the file ending is not supported.

        """
        loader = self._get_document_loader(directory, file_ending)
        docs = self._load_and_split_documents(loader)
        self._embed_and_store_documents(docs)

    def _get_document_loader(self, directory: str, file_ending: str) -> DirectoryLoader:
        """Get the appropriate document loader based on file ending."""
        if file_ending == ".pdf":
            return DirectoryLoader(directory, glob=f"*{file_ending}", loader_cls=PyPDFium2Loader)
        elif file_ending == ".txt":
            return DirectoryLoader(directory, glob=f"*{file_ending}", loader_cls=TextLoader)
        else:
            msg = "File ending not supported."
            raise ValueError(msg)

    def _load_and_split_documents(self, loader: DirectoryLoader) -> list[Any]:
        """Load and split documents using the NLTK text splitter."""
        splitter = NLTKTextSplitter(length_function=len, chunk_size=500, chunk_overlap=75)
        return loader.load_and_split(splitter)

    def _embed_and_store_documents(self, docs: list[Any]) -> None:
        """Embed and store documents in the vector database."""
        logger.info(f"Loaded {len(docs)} documents.")
        text_list = [doc.page_content for doc in docs]
        metadata_list = [self._process_metadata(doc.metadata) for doc in docs]

        self.vector_db.add_texts(texts=text_list, metadatas=metadata_list)
        logger.info("SUCCESS: Texts embedded.")

    @staticmethod
    def _process_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Process metadata to extract the filename from the source path."""
        if "source" in metadata and "/" in metadata["source"]:
            metadata["source"] = metadata["source"].split("/")[-1]
        return metadata

    def summarize_text(self, text: str) -> str:
        """Summarize the given text using the OpenAI API.

        Args:
        ----
            text (str): The text to be summarized.

        Returns:
        -------
            str: The summary of the text.

        """
        prompt = load_prompt_template(prompt_name="openai-summarization.j2", text=text, language="de")

        response = openai.chat.completions.create(
            model=self.cfg.openai_completion.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.openai_completion.temperature,
            max_tokens=self.cfg.openai_completion.max_tokens,
            top_p=self.cfg.openai_completion.top_p,
            frequency_penalty=self.cfg.openai_completion.frequency_penalty,
            presence_penalty=self.cfg.openai_completion.presence_penalty,
            stop=self.cfg.openai_completion.stop,
            stream=False,
        )

        return response.choices[0].messages.content

    def create_search_chain(self, search: SearchParams) -> list[dict]:
        pass

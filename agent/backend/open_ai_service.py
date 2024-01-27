"""This script is used to initialize the Qdrant db backend with Azure OpenAI."""
import os
from typing import Any

import openai
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain_community.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.utils.utility import generate_prompt
from agent.utils.vdb import init_vdb

load_dotenv()


@load_config(location="config/db.yml")
def get_db_connection(open_ai_token: str, cfg: DictConfig, collection_name: str) -> Qdrant:
    """Initializes a connection to the Qdrant DB.

    Args:
    ----
        open_ai_token (str): The openai token.
        cfg (DictConfig): the config file.
        collection_name (str): The name of the vector database collection.

    Returns:
    -------
        Qdrant: An Langchain Instance of the Qdrant DB.
    """
    if cfg.openai.azure:
        embedding = AzureOpenAIEmbeddings(deployment=cfg.openai.deployment, openai_api_version="2023-05-15", openai_api_key=open_ai_token)  # type: ignore
    else:
        embedding = OpenAIEmbeddings(model=cfg.openai.deployment, openai_api_key=open_ai_token)

    if collection_name is None or not collection_name:
        collection_name = cfg.qdrant.collection_name_openai

    return init_vdb(cfg, collection_name, embedding)


def embedd_documents_openai(dir: str, open_ai_token: str, collection_name: str | None = None) -> None:
    """embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    :param open_ai_token: OpenAI API Token
    :type open_ai_token: str
    """
    vector_db: Qdrant = get_db_connection(open_ai_token=open_ai_token, collection_name=collection_name)

    splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFium2Loader)
    docs = loader.load_and_split(splitter)

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")


def search_documents_openai(open_ai_token: str, query: str, amount: int, threshold: float = 0.0, collection_name: str | None = None) -> list[tuple[Document, float]]:
    """Searches the documents in the Qdrant DB with a specific query.

    Args:
    ----
        open_ai_token (str): The OpenAI API token.
        query (str): The question for which documents should be searched.

    Returns:
    -------
        List[Tuple[Document, float]]: A list of search results, where each result is a tuple
        containing a Document object and a float score.
    """
    vector_db = get_db_connection(open_ai_token=open_ai_token, collection_name=collection_name)

    docs = vector_db.similarity_search_with_score(query, k=amount, score_threshold=threshold)
    logger.info("SUCCESS: Documents found.")
    return docs


@load_config(location="config/ai/openai.yml")
def summarize_text_openai(text: str, token: str, cfg: DictConfig) -> str:
    """Summarizes the given text using the Luminous API.

    Args:
    ----
        text (str): The text to be summarized.
        token (str): The token for the Luminous API.

    Returns:
    -------
        str: The summary of the text.
    """
    prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, language="de")

    openai.api_key = token
    response = openai.Completion.create(
        engine=cfg.openai.model,
        prompt=prompt,
        temperature=cfg.openai.temperature,
        max_tokens=cfg.openai.max_tokens,
        top_p=cfg.openai.top_p,
        frequency_penalty=cfg.openai.frequency_penalty,
        presence_penalty=cfg.openai.presence_penalty,
        best_of=cfg.openai.best_of,
        stop=cfg.openai.stop,
    )

    return response.choices[0].text


@load_config(location="config/ai/openai.yml")
def send_completion(text: str, query: str, token: str, cfg: DictConfig) -> str:
    """Sent completion request to OpenAI API.

    Args:
    ----
        text (str): The text on which the completion should be based.
        query (str): The query for the completion.
        token (str): The token for the OpenAI API.
        cfg (DictConfig):

    Returns:
    -------
        str: Response from the OpenAI API.
    """
    prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, query=query, language="de")

    openai.api_key = token
    response = openai.Completion.create(
        engine=cfg.openai.model,
        prompt=prompt,
        temperature=cfg.openai.temperature,
        max_tokens=cfg.openai.max_tokens,
        top_p=cfg.openai.top_p,
        frequency_penalty=cfg.openai.frequency_penalty,
        presence_penalty=cfg.openai.presence_penalty,
        best_of=cfg.openai.best_of,
        stop=cfg.openai.stop,
    )

    return response.choices[0].text


def send_custom_completion_openai(
    token: str,
    prompt: str,
    model: str = "gpt3.5",
    max_tokens: int = 256,
    stop_sequences: list[str] | None = None,
    temperature: float = 0,
) -> str:
    """Sent completion request to OpenAI API.

    Args:
    ----
        text (str): The text on which the completion should be based.
        query (str): The query for the completion.
        token (str): The token for the OpenAI API.
        cfg (DictConfig):

    Returns:
    -------
        str: Response from the OpenAI API.
    """
    if stop_sequences is None:
        stop_sequences = ["###"]
    openai.api_key = token
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
    )

    return response.choices[0].text


def qa_openai(token: str, documents: list[tuple[Document, float]], query: str, summarization: bool = False) -> tuple[Any, str, dict[Any, Any]]:
    """QA Function for OpenAI LLMs.

    Args:
    ----
        token (str): The token for the OpenAI API.
        documents (list[tuple[Document, float]]): The documents to be searched.
        query (str): The question for which the LLM should generate an answer.
        summarization (bool, optional): If the Documents should be summarized. Defaults to False.

    Returns:
    -------
        tuple: answer, prompt, meta_data
    """
    # if the list of documents contains only one document extract the text directly
    if len(documents) == 1:
        text = documents[0][0].page_content
        meta_data = documents[0][0].metadata

    else:
        # extract the text from the documents
        texts = [doc[0].page_content for doc in documents]
        if summarization:
            # call summarization
            text = ""
            for t in texts:
                text += summarize_text_openai(text=t, token=token)

        else:
            # combine the texts to one text
            text = " ".join(texts)
        meta_data = [doc[0].metadata for doc in documents]

    # load the prompt
    prompt = generate_prompt("aleph_alpha_qa.j2", text=text, query=query)

    try:
        # call the luminous api
        answer = send_completion(prompt, token)

    except ValueError as e:
        # if the code is PROMPT_TOO_LONG, split it into chunks
        if e.args[0] == "PROMPT_TOO_LONG":
            logger.info("Prompt too long. Summarizing.")

            # summarize the text
            short_text = summarize_text_openai(text, token)

            # generate the prompt
            prompt = generate_prompt("openai-qa.j2", text=short_text, query=query)

            # call the luminous api
            answer = send_completion(prompt, token)

    # extract the answer
    return answer, prompt, meta_data


if __name__ == "__main__":
    token = os.getenv("OPENAI_API_KEY")

    if not token:
        msg = "OPENAI_API_KEY is not set."
        raise ValueError(msg)

    embedd_documents_openai(dir="data", open_ai_token=token)

    DOCS = search_documents_openai(open_ai_token="", query="Was ist Vanille?", amount=3)

    summary = summarize_text_openai(text="Below is an extract from the annual financial report of a company. ", token=token)

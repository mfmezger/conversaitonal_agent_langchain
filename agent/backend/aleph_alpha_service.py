"""The script to initialize the Qdrant db backend with aleph alpha."""

import os
from pathlib import Path
from typing import Any

import nltk
import numpy as np
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    Document,
    ExplanationRequest,
    Prompt,
    SummarizationRequest,
)
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain_community.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.utils.utility import generate_prompt
from agent.utils.vdb import init_vdb

nltk.download("punkt")  # This needs to be installed for the tokenizer to work.
load_dotenv()

aleph_alpha_token = os.getenv("ALEPH_ALPHA_API_KEY")
tokenizer = None


def get_tokenizer() -> None:
    """Initialize the tokenizer."""
    global tokenizer
    client = Client(token=aleph_alpha_token)
    tokenizer = client.tokenizer("luminous-base")


def count_tokens(text: str):
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


@load_config(location="config/db.yml")
def get_db_connection(aleph_alpha_token: str, cfg: DictConfig, collection_name: str | None = None) -> Qdrant:
    """Initializes a connection to the Qdrant DB.

    Args:
    ----
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
    -------
        Qdrant: The Qdrant DB connection.
    """
    embedding = AlephAlphaAsymmetricSemanticEmbedding(
        model=cfg.aleph_alpha_embeddings.model_name,
        aleph_alpha_api_key=aleph_alpha_token,
        normalize=cfg.aleph_alpha_embeddings.normalize,
        compress_to_size=cfg.aleph_alpha_embeddings.compress_to_size,
    )

    if collection_name is None or not collection_name:
        collection_name = cfg.qdrant.collection_name_aa

    return init_vdb(cfg, collection_name, embedding)


def summarize_text_aleph_alpha(text: str, token: str) -> str:
    """Summarizes the given text using the Luminous API.

    Args:
    ----
        text (str): The text to be summarized.
        token (str): The token for the Luminous API.

    Returns:
    -------
        str: The summary of the text.
    """
    client = Client(token=token)
    document = Document.from_text(text=text)
    request = SummarizationRequest(document=document)
    response = client.summarize(request=request)

    return response.summary


@load_config(location="config/ai/aleph_alpha.yml")
def send_completion_request(text: str, token: str, cfg: DictConfig) -> str:
    """Sends a completion request to the Luminous API.

    Args:
    ----
        text (str): The prompt to be sent to the API.
        token (str): The token for the Luminous API.

    Returns:
    -------
        str: The response from the API.

    Raises:
    ------
        ValueError: If the text or token is None or empty, or if the response or completion is empty.
    """
    if not text:
        msg = "Text cannot be None or empty."
        raise ValueError(msg)
    if not token:
        msg = "Token cannot be None or empty."
        raise ValueError(msg)

    client = Client(token=token)

    request = CompletionRequest(
        prompt=Prompt.from_text(text),
        maximum_tokens=cfg.aleph_alpha_completion.max_tokens,
        stop_sequences=[cfg.aleph_alpha_completion.stop_sequences],
        repetition_penalties_include_completion=cfg.aleph_alpha_completion.repetition_penalties_include_completion,
    )
    response = client.complete(request, model=cfg.aleph_alpha_completion.model)

    # ensure that the response is not empty
    if not response.completions:
        msg = "Response is empty."
        raise ValueError(msg)

    # ensure that the completion is not empty
    if not response.completions[0].completion:
        msg = "Completion is empty."
        raise ValueError(msg)

    return str(response.completions[0].completion)


def embedd_documents_aleph_alpha(dir: str, aleph_alpha_token: str, collection_name: str | None = None) -> None:
    """Embeds the documents in the given directory in the Aleph Alpha database.

    This method uses the Directory Loader for PDFs and the PyPDFium2Loader to load the documents.
    The documents are then added to the Qdrant DB which embeds them without deleting the old collection.

    Args:
    ----
        dir (str): The directory containing the PDFs to embed.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
    -------
        None
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name, aleph_alpha_token=aleph_alpha_token)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFium2Loader)
    get_tokenizer()

    splitter = NLTKTextSplitter(length_function=count_tokens, chunk_size=300, chunk_overlap=50)
    docs = loader.load_and_split(splitter)

    logger.info(f"Loaded {len(docs)} documents.")
    text_list = [doc.page_content for doc in docs]
    metadata_list = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=text_list, metadatas=metadata_list)

    logger.info("SUCCESS: Texts embedded.")


def embedd_text_aleph_alpha(text: str, file_name: str, aleph_alpha_token: str, seperator: str, collection_name: str | None = None) -> None:
    """Embeds the given text in the Aleph Alpha database.

    Args:
    ----
        text (str): The text to be embedded.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
    -------
        None
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name, aleph_alpha_token=aleph_alpha_token)

    # split the text at the seperator
    text_list: list = text.split(seperator)

    # check if first and last element are empty
    if not text_list[0]:
        text_list.pop(0)
    if not text_list[-1]:
        text_list.pop(-1)

    metadata = file_name
    # add _ and an incrementing number to the metadata
    metadata_list: list = [{"source": f"{metadata}_{i!s}", "page": 0} for i in range(len(text_list))]

    vector_db.add_texts(texts=text_list, metadatas=metadata_list)
    logger.info("SUCCESS: Text embedded.")


def embedd_text_files_aleph_alpha(folder: str, aleph_alpha_token: str, seperator: str, collection_name: str | None = None) -> None:
    """Embeds text files in the Aleph Alpha database.

    Args:
    ----
        folder (str): The folder containing the text files to embed.
        aleph_alpha_token (str): The Aleph Alpha API token.
        seperator (str): The seperator to use when splitting the text into chunks.

    Returns:
    -------
        None
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name, aleph_alpha_token=aleph_alpha_token)

    # iterate over the files in the folder
    for file in os.listdir(folder):
        # check if the file is a .txt or .md file
        if not file.endswith((".txt", ".md")):
            continue

        # read the text from the file
        text = Path(Path(folder) / file).read_text()
        text_list: list = text.split(seperator)

        # check if first and last element are empty
        if not text_list[0]:
            text_list.pop(0)
        if not text_list[-1]:
            text_list.pop(-1)

        # ensure that the text is not empty
        if not text_list:
            msg = "Text is empty."
            raise ValueError(msg)

        logger.info(f"Loaded {len(text_list)} documents.")
        # get the name of the file
        metadata = os.path.splitext(file)[0]
        # add _ and an incrementing number to the metadata
        metadata_list: list = [{"source": f"{metadata}_{i!s}", "page": 0} for i in range(len(text_list))]
        vector_db.add_texts(texts=text_list, metadatas=metadata_list)

    logger.info("SUCCESS: Text embedded.")


def search_documents_aleph_alpha(
    aleph_alpha_token: str, query: str, amount: int = 1, threshold: float = 0.0, collection_name: str | None = None
) -> list[tuple[LangchainDocument, float]]:
    """Searches the Aleph Alpha service for similar documents.

    Args:
    ----
        aleph_alpha_token (str): Aleph Alpha API Token.
        query (str): The query that should be searched for.
        amount (int, optional): The number of documents to return. Defaults to 1.

    Returns:
    -------
        List[Tuple[Document, float]]: A list of tuples containing the documents and their similarity scores.
    """
    if not aleph_alpha_token:
        msg = "Token cannot be None or empty."
        raise ValueError(msg)
    if not query:
        msg = "Query cannot be None or empty."
        raise ValueError(msg)
    if amount < 1:
        msg = "Amount must be greater than 0."
        raise ValueError(msg)
    # TODO: FILTER
    try:
        vector_db: Qdrant = get_db_connection(collection_name=collection_name, aleph_alpha_token=aleph_alpha_token)
        docs = vector_db.similarity_search_with_score(query=query, k=amount, score_threshold=threshold)
        logger.info("SUCCESS: Documents found.")
        return docs
    except Exception as e:
        logger.error(f"ERROR: Failed to search documents: {e}")
        msg = f"Failed to search documents: {e}"
        raise Exception(msg) from e


def qa_aleph_alpha(
    aleph_alpha_token: str, documents: list[tuple[LangchainDocument, float]], query: str, summarization: bool = False
) -> tuple[str, str, dict[Any, Any] | list[dict[Any, Any]]]:
    """QA takes a list of documents and returns a list of answers.

    Args:
    ----
        aleph_alpha_token (str): The Aleph Alpha API token.
        documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
        query (str): The query to ask.
        summarization (bool, optional): Whether to use summarization. Defaults to False.

    Returns:
    -------
        Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
    """
    # if the list of documents contains only one document extract the text directly
    if len(documents) == 1:
        text = documents[0][0].page_content
        meta_data = documents[0][0].metadata

    else:
        # extract the text from the documents
        texts = [doc[0].page_content for doc in documents]
        if summarization:
            text = "".join(summarize_text_aleph_alpha(t, aleph_alpha_token) for t in texts)
        else:
            # combine the texts to one text
            text = " ".join(texts)
        meta_data = [doc[0].metadata for doc in documents]

    # load the prompt
    prompt = generate_prompt("aleph_alpha_qa.j2", text=text, query=query)

    try:
        # call the luminous api
        answer = send_completion_request(prompt, aleph_alpha_token)

    except ValueError as e:
        # if the code is PROMPT_TOO_LONG, split it into chunks
        if e.args[0] == "PROMPT_TOO_LONG":
            logger.info("Prompt too long. Summarizing.")

            # summarize the text
            short_text = summarize_text_aleph_alpha(text, aleph_alpha_token)

            # generate the prompt
            prompt = generate_prompt("aleph_alpha_qa.j2", text=short_text, query=query)

            # call the luminous api
            answer = send_completion_request(prompt, aleph_alpha_token)

    # extract the answer
    return answer, prompt, meta_data


@load_config(location="config/ai/aleph_alpha.yml")
def explain_qa(aleph_alpha_token: str, document: LangchainDocument, explain_threshold: float, query: str, cfg: DictConfig, collection_name: str | None = None) -> tuple:
    """Explian QA WIP."""
    text = document[0][0].page_content
    meta_data = document[0][0].metadata

    # load the prompt
    prompt = generate_prompt("aleph_alpha_qa.j2", text=text, query=query)

    answer = send_completion_request(prompt, aleph_alpha_token)

    exp_req = ExplanationRequest(Prompt.from_text(prompt), answer, control_factor=0.1, prompt_granularity="sentence", normalize=True)
    client = Client(token=aleph_alpha_token)

    response_explain = client.explain(exp_req, model=cfg.aleph_alpha_completion.model)
    explanations = response_explain.explanations[0].items[0].scores

    # if all of the scores are belo 0.7 raise an error
    if all(item.score < explain_threshold for item in explanations):
        msg = "All scores are below explain_threshold."
        raise ValueError(msg)

    # remove element if the text contains Response: or Instructions:
    for exp in explanations:
        txt = prompt[exp.start : exp.start + exp.length]
        if "Response:" in txt or "Instruction:" in txt:
            explanations.remove(exp)

    # pick the top explanation based on score
    top_explanation = max(explanations, key=lambda x: x.score)

    # get the start and end of the explanation
    start = top_explanation.start
    end = top_explanation.start + top_explanation.length

    # get the explanation from the prompt
    explanation = prompt[start:end]

    # get the score
    score = np.round(top_explanation.score, decimals=3)

    # get the text from the document
    text = document[0][0].page_content

    return explanation, score, text, answer, meta_data


def explain_completion(prompt: str, output: str, token: str) -> dict[str, float]:
    # TODO: repair
    """Returns an explanation of the given completion.

    Args:
    ----
        prompt (str): The complete input in the model.
        output (str): The answer of the model.
        token (str): The Aleph Alpha API Token.

    Returns:
    -------
        dict: A dictionary containing the explanation. The keys are sentences from the prompt, and the values are the scores.

    Raises:
    ------
        ValueError: If the prompt, output, or token is None or empty.
    """
    exp_req = ExplanationRequest(Prompt.from_text(prompt), output, control_factor=0.1, prompt_granularity="sentence")
    client = Client(token=token)
    response_explain = client.explain(exp_req, model="luminous-extended-control")
    explanations = response_explain.explanations[0].items[0]

    # sort the explanations by score
    # explanations = sorted(explanations, key=lambda x: x.score, reverse=True)

    template = generate_prompt(prompt_name="aleph_alpha_qa.j2", text="", language="de")

    result = {}
    # remove the prompt from the explanations
    for item in explanations:
        start = item.start
        end = item.start + item.length
        if prompt[start:end] not in template:
            result[prompt[start:end]] = np.round(item.score, decimals=3)

    return result


def process_documents_aleph_alpha(folder: str, token: str, type: str) -> list[str]:
    """Process the documents in the given folder.

    Args:
    ----
        folder (str): Folder where the documents are located.
        token (str): The Aleph Alpha API Token.
        type (str): The type of the documents.

    Raises:
    ------
        ValueError: If the type is not one of 'qa', 'summarization', or 'invoice'.
    """
    # load the documents
    loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFium2Loader)

    # load the documents
    docs = loader.load()

    # load the correct prompt
    match type:
        case "qa":
            raise NotImplementedError
        case "summarization":
            raise NotImplementedError
        case "invoice":
            # load the prompt
            prompt_name = "aleph-alpha-invoice.j2"
        case _:
            msg = "Type must be one of 'qa', 'summarization', or 'invoice'."
            raise ValueError(msg)

    # generate the prompt
    answers = []
    # iterate over the documents
    for doc in docs:
        # combine the prompt and the text
        prompt_text = generate_prompt(prompt_name=prompt_name, text=doc.page_content, language="en")
        # call the luminous api
        answer = send_completion_request(prompt_text, token)

        answers.append(answer)

    return answers


def custom_completion_prompt_aleph_alpha(
    token: str,
    prompt: str,
    model: str = "luminous-extended-control",
    max_tokens: int = 256,
    stop_sequences: list[str] | None = None,
    temperature: float = 0,
) -> str:
    """This method sents a custom completion request to the Aleph Alpha API.

    Args:
    ----
        token (str): The token for the Aleph Alpha API.
        prompt (str): The prompt to be sent to the API.

    Raises:
    ------
        ValueError: Error if their are no completions or the completion is empty or the prompt and tokenis empty.
    """
    if stop_sequences is None:
        stop_sequences = ["###"]
    if not prompt:
        msg = "Prompt cannot be None or empty."
        raise ValueError(msg)
    if not token:
        msg = "Token cannot be None or empty."
        raise ValueError(msg)

    client = Client(token=token)

    request = CompletionRequest(prompt=Prompt.from_text(prompt), maximum_tokens=max_tokens, stop_sequences=stop_sequences, temperature=temperature)
    response = client.complete(request, model=model)

    # ensure that the response is not empty
    if not response.completions:
        msg = "Response is empty."
        raise ValueError(msg)

    # ensure that the completion is not empty
    if not response.completions[0].completion:
        msg = "Completion is empty."
        raise ValueError(msg)

    return str(response.completions[0].completion)


def qa_chain(query: str, token: str, collection_name: str) -> None:
    """QA Chain Impl."""
    # TODO:
    vector_db: Qdrant = get_db_connection(collection_name=collection_name, aleph_alpha_token=token)

    retriever = vector_db.as_retriever()

    retriever.invoke(query)


def self_question_qa() -> None:
    """Self question QA."""
    # TODO


if __name__ == "__main__":
    token = os.getenv("ALEPH_ALPHA_API_KEY")

    if not token:
        msg = "Token cannot be None or empty."
        raise ValueError(msg)

    embedd_documents_aleph_alpha("data", token)
    # open the text file and read the text
    DOCS = search_documents_aleph_alpha(aleph_alpha_token=token, query="What are Attentions?", amount=1)
    logger.info(DOCS)
    # explanation, score, text, answer, meta_data = explain_qa(aleph_alpha_token=token, document=DOCS, query="What are Attentions?")
    # logger.info(f"Answer: {answer}")
    # explanations = explain_completion(prompt, answer, token)

    # print(explanation)

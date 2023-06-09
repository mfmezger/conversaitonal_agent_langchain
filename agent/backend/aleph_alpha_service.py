"""The script to initialize the chroma db backend with aleph alpha."""
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from aleph_alpha_client import Client, CompletionRequest, ExplanationRequest, Prompt
from dotenv import load_dotenv
from jinja2 import Template
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain.vectorstores import Chroma
from loguru import logger
from omegaconf import DictConfig

from agent.utils.configuration import load_config

load_dotenv()


def generate_prompt(prompt_name: str, text: str, query: str) -> str:
    """Generates a prompt for the Luminous API using a Jinja template.

    Args:
        prompt_name (str): The name of the file containing the Jinja template.
        text (str): The text to be inserted into the template.
        query (str): The query to be inserted into the template.

    Returns:
        str: The generated prompt.

    Raises:
        FileNotFoundError: If the specified prompt file cannot be found.
    """
    try:
        with open(os.path.join("prompts", prompt_name)) as f:
            prompt = Template(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_name}' not found.")

    # replace the value text with jinja
    # Render the template with your variable
    prompt = prompt.render(text=text, query=query)

    return str(prompt)


def send_completion_request(text: str, token: str) -> str:
    """Sends a completion request to the Luminous API.

    Args:
        text (str): The prompt to be sent to the API.
        token (str): The token for the Luminous API.

    Returns:
        str: The response from the API.

    Raises:
        ValueError: If the text or token is None or empty, or if the response or completion is empty.
    """
    if not text:
        raise ValueError("Text cannot be None or empty.")
    if not token:
        raise ValueError("Token cannot be None or empty.")

    client = Client(token=token)

    request = CompletionRequest(prompt=Prompt.from_text(text), maximum_tokens=256, stop_sequences=["###"])
    response = client.complete(request, model="luminous-supreme-control")

    # ensure that the response is not empty
    if not response.completions:
        raise ValueError("Response is empty.")

    # ensure that the completion is not empty
    if not response.completions[0].completion:
        raise ValueError("Completion is empty.")

    return response.completions[0].completion


@load_config(location="config/chroma_db.yml")
def get_db_connection(cfg: DictConfig, aleph_alpha_token: str) -> Chroma:
    """Initializes a connection to the Chroma DB.

    Args:
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        Chroma: The Chroma DB connection.
    """
    embedding = AlephAlphaAsymmetricSemanticEmbedding(aleph_alpha_api_key=aleph_alpha_token)
    vector_db = Chroma(persist_directory=cfg.chroma.persist_directory_aa, embedding_function=embedding)

    logger.info("SUCCESS: Chroma DB initialized.")

    return vector_db


def embedd_documents_aleph_alpha(dir: str, aleph_alpha_token: str) -> None:
    """Embeds the documents in the given directory in the Aleph Alpha database.

    This method uses the Directory Loader for PDFs and the PyPDFLoader to load the documents.
    The documents are then added to the Chroma DB which embeds them without deleting the old collection.

    Args:
        dir (str): The directory containing the PDFs to embed.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        None
    """
    vector_db = get_db_connection(aleph_alpha_token=aleph_alpha_token)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")
    vector_db.persist()
    logger.info("SUCCESS: Database Persistent.")


def embedd_text_aleph_alpha(text: str, file_name: str, aleph_alpha_token: str, seperator: str) -> None:
    """Embeds the given text in the Aleph Alpha database.

    Args:
        text (str): The text to be embedded.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        None
    """
    vector_db = get_db_connection(aleph_alpha_token=aleph_alpha_token)

    # split the text at the seperator
    text_list: List = text.split(seperator)

    metadata = file_name
    # add _ and an incrementing number to the metadata
    metadata_list: List = [metadata + "_" + str(i) for i in range(len(text_list))]

    vector_db.add_texts(texts=text_list, metadata=metadata_list)
    logger.info("SUCCESS: Text embedded.")
    vector_db.persist()
    logger.info("SUCCESS: Database Persistent.")


def embedd_text_files_aleph_alpha(folder: str, aleph_alpha_token: str, separator: str) -> None:
    """Embeds text files in the Aleph Alpha database.

    Args:
        folder (str): The folder containing the text files to embed.
        aleph_alpha_token (str): The Aleph Alpha API token.
        separator (str): The separator to use when splitting the text into chunks.

    Returns:
        None
    """
    vector_db = get_db_connection(aleph_alpha_token=aleph_alpha_token)

    # iterate over the files in the folder
    for file in os.listdir(folder):
        # check if the file is a .txt or .md file
        if not file.endswith((".txt", ".md")):
            continue

        # read the text from the file
        with open(os.path.join(folder, file)) as f:
            text = f.read()

        text_list: List = text.split(separator)

        # get the name of the file
        metadata = os.path.splitext(file)[0]
        # add _ and an incrementing number to the metadata
        metadata_list: List = [metadata + "_" + str(i) for i in range(len(text_list))]

        vector_db.add_texts(texts=text_list, metadata=metadata_list)

    logger.info("SUCCESS: Text embedded.")
    vector_db.persist()
    logger.info("SUCCESS: Database Persistent.")


def search_documents_aleph_alpha(aleph_alpha_token: str, query: str, amount: int = 1) -> List[Tuple[Document, float]]:
    """Searches the Aleph Alpha service for similar documents.

    Args:
        aleph_alpha_token (str): Aleph Alpha API Token.
        query (str): The query that should be searched for.
        amount (int, optional): The number of documents to return. Defaults to 1.

    Returns
        List[Tuple[Document, float]]: A list of tuples containing the documents and their similarity scores.
    """
    if not aleph_alpha_token:
        raise ValueError("Token cannot be None or empty.")
    if not query:
        raise ValueError("Query cannot be None or empty.")
    if amount < 1:
        raise ValueError("Amount must be greater than 0.")

    try:
        vector_db = get_db_connection(aleph_alpha_token=aleph_alpha_token)
        docs = vector_db.similarity_search_with_score(query, k=amount)
        logger.info("SUCCESS: Documents found.")
        return docs
    except Exception as e:
        logger.error(f"ERROR: Failed to search documents: {e}")
        raise


# def summarization(aleph_alpha_token: str, documents: List[Tuple[Document, float]]) -> List[str]:
#     """Summarizes a list of documents and returns a list of summaries.

#     Args:
#         aleph_alpha_token (str): Aleph Alpha API Token.
#         documents (List[Tuple[Document, float]]): A list of tuples containing the documents and their similarity scores.

#     Returns
#     -------
#         List[str]: A list of summaries.

#     """
#     client = Client(token=aleph_alpha_token)

#     # TODO: Implement
#     # extract the text from the documents
#     texts = [doc[0].page_content for doc in documents]

#     # summarize the texts
#     summaries = []
#     for text in texts:
#         pass

#     return ["None"]


def qa_aleph_alpha(
    aleph_alpha_token: str, documents: List[Tuple[Document, float]], query: str, summarization: bool = False
) -> Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
    """QA takes a list of documents and returns a list of answers.

    Args:
        aleph_alpha_token (str): The Aleph Alpha API token.
        documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
        query (str): The query to ask.
        summarization (bool, optional): Whether to use summarization. Defaults to False.

    Returns:
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
            # call summarization
            # TODO: implement summarization
            text = ""
        else:
            # combine the texts to one text
            text = " ".join(texts)
        meta_data = [doc[0].metadata for doc in documents]

    # load the prompt
    prompt = generate_prompt("qa.j2", text=text, query=query)
    # call the luminous api
    answer = send_completion_request(prompt, aleph_alpha_token)

    # extract the answer
    return answer, prompt, meta_data


def explain_completion(prompt: str, output: str, token: str):
    """Returns an explanation of the given completion.

    Args:
        prompt (str): The complete input in the model.
        output (str): The answer of the model.
        token (str): The Aleph Alpha API Token.

    Returns:
        dict: A dictionary containing the explanation. The keys are sentences from the prompt, and the values are the scores.

    Raises:
        ValueError: If the prompt, output, or token is None or empty.
    """
    exp_req = ExplanationRequest(Prompt.from_text(prompt), output, control_factor=0.1, prompt_granularity="sentence")
    client = Client(token=token)
    response_explain = client.explain(exp_req, model="luminous-supreme-control")

    explanations = response_explain[1][0].items[0][0]

    # sort the explanations by score
    explanations = sorted(explanations, key=lambda x: x.score, reverse=True)

    result = {}
    # extract the first 3 explanations
    for item in explanations[:3]:
        start = item.start
        end = item.start + item.length
        result[prompt[start:end]] = np.round(item.score, decimals=3)

    return result


if __name__ == "__main__":
    # embedd_documents_aleph_alpha("data", str(os.getenv("ALEPH_ALPHA_API_KEY")))
    # open the text file and read the text
    with open("data/brustkrebs_input.txt") as f:
        text = f.read()

    # embedd_text_aleph_alpha(text, "file1", str(os.getenv("ALEPH_ALPHA_API_KEY")), "###")
    embedd_text_files_aleph_alpha("data/", str(os.getenv("ALEPH_ALPHA_API_KEY")), "###")
    DOCS = search_documents_aleph_alpha(aleph_alpha_token=str(os.getenv("ALEPH_ALPHA_API_KEY")), query="Was sind meine Vorteile?")
    logger.info(DOCS)
    answer, prompt, meta_data = qa_aleph_alpha(aleph_alpha_token=str(os.getenv("ALEPH_ALPHA_API_KEY")), documents=DOCS, query="Muss ich mein Mietwagen volltanken?")
    logger.info(f"Answer: {answer}")
    explanations = explain_completion(prompt, answer, str(os.getenv("ALEPH_ALPHA_API_KEY")))

    print(explanations)

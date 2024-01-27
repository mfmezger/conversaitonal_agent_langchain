"""GPT4ALL Backend Service."""

from dotenv import load_dotenv
from gpt4all import GPT4All
from langchain.docstore.document import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from agent.utils.utility import generate_prompt
from agent.utils.vdb import init_vdb

load_dotenv()


@load_config(location="config/db.yml")
def get_db_connection(cfg: DictConfig, collection_name: str) -> Qdrant:
    """Initializes a connection to the Qdrant DB.

    Args:
    ----
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
    -------
        Qdrant: The Qdrant DB connection.
    """
    embedding = GPT4AllEmbeddings()
    if collection_name is None or not collection_name:
        collection_name = cfg.qdrant.collection_name_gpt4all

    return init_vdb(cfg, collection_name, embedding)


def embedd_documents_gpt4all(dir: str, collection_name: str | None = None) -> None:
    """embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name)

    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFium2Loader)
    splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = loader.load_and_split(splitter)

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=texts, metadatas=metadatas)
    logger.info("SUCCESS: Texts embedded.")


def embedd_text_gpt4all(text: str, file_name: str, seperator: str, collection_name: str | None = None) -> None:
    """embedd_documents embedds the documents in the given directory.

    :param cfg: Configuration from the file
    :type cfg: DictConfig
    :param dir: PDF Directory
    :type dir: str
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name)

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


@load_config(location="config/ai/gpt4all.yml")
def summarize_text_gpt4all(text: str, cfg: DictConfig) -> str:
    """Summarize text with GPT4ALL.

    Args:
    ----
        text (str): The text to be summarized.

    Returns:
    -------
        str: The summarized text.
    """
    prompt = generate_prompt(prompt_name="openai-summarization.j2", text=text, language="de")

    model = GPT4All(cfg.gpt4all.completion_model)

    return model.generate(prompt, max_tokens=300)


@load_config(location="config/ai/gpt4all.yml")
def completion_text_gpt4all(prompt: str, cfg: DictConfig) -> str:
    """Complete text with GPT4ALL.

    Args:
    ----
        text (str): The text as basic input.
        query (str): The query to be inserted into the template.

    Returns:
    -------
        str: The completed text.
    """
    model = GPT4All(cfg.gpt4all.completion_model)

    return model.generate(prompt, max_tokens=100)


def custom_completion_prompt_gpt4all(prompt: str, model: str = "orca-mini-3b.ggmlv3.q4_0.bin", max_tokens: int = 256, temperature: float = 0) -> str:
    """This method sents a custom completion request to the Aleph Alpha API.

    Args:
    ----
        token (str): The token for the Aleph Alpha API.
        prompt (str): The prompt to be sent to the API.

    Raises:
    ------
        ValueError: Error if their are no completions or the completion is empty or the prompt and tokenis empty.
    """
    if not prompt:
        msg = "Prompt cannot be None or empty."
        raise ValueError(msg)

    output = (GPT4All(model)).generate(prompt, max_tokens=max_tokens, temp=temperature)

    return str(output)


def search_documents_gpt4all(query: str, amount: int, threshold: float = 0.0, collection_name: str | None = None) -> list[tuple[Document, float]]:
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
    vector_db: Qdrant = get_db_connection(collection_name=collection_name)

    docs = vector_db.similarity_search_with_score(query=query, k=amount, score_threshold=threshold)
    logger.info("SUCCESS: Documents found.")
    return docs


def qa_gpt4all(documents: list[tuple[Document, float]], query: str, summarization: bool = False, language: str = "de"):
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
            text = "".join(summarize_text_gpt4all(t) for t in texts)
        else:
            # combine the texts to one text
            text = " ".join(texts)
        meta_data = [doc[0].metadata for doc in documents]

    # load the prompt
    prompt = generate_prompt("gpt4all-completion.j2", text=text, query=query, language=language)

    try:
        # call the luminous api
        logger.info("starting completion")
        answer = completion_text_gpt4all(prompt)
        logger.info(f"completion done with answer {answer}")

    except ValueError as e:
        # if the code is PROMPT_TOO_LONG, split it into chunks
        if e.args[0] == "PROMPT_TOO_LONG":
            logger.info("Prompt too long. Summarizing.")

            # summarize the text
            short_text = summarize_text_gpt4all(text)

            # generate the prompt
            prompt = generate_prompt("gpt4all-completion.j2", text=short_text, query=query, language=language)

            # call the luminous api
            answer = completion_text_gpt4all(prompt)

    # extract the answer
    return answer, prompt, meta_data


if __name__ == "__main__":
    embedd_documents_gpt4all(dir="data")

    # print(f'Summary: {summarize_text_gpt4all(text="Das ist ein Test.")}')

    # print(f'Completion: {completion_text_gpt4all(text="Das ist ein Test.", query="Was ist das?")}')

    answer, prompt, meta_data = qa_gpt4all(documents=search_documents_gpt4all(query="Das ist ein Test.", amount=1), query="Was ist das?")

    logger.info(f"Answer: {answer}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Metadata: {meta_data}")

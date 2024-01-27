"""The script to initialize a vector db in the qdrant database server and ingested custom data with aleph alpha embeddings."""
import json
import os
import re
from pathlib import Path

from aleph_alpha_client import Client
from dotenv import load_dotenv
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain_community.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from ultra_simple_config import load_config

load_dotenv()


@load_config(location="config/db.yml")
def setup_tokenizer_client(cfg: DictConfig):
    """Set up the tokenizer and the aleph alpha client.

    Args:
    ----
        cfg (DictConfig): The config data for the embeddings.

    Returns:
    -------
        client, tokenizer: the aleph alpha client and the tokenizer
    """
    client = Client(token=aleph_alpha_token)
    tokenizer = client.tokenizer(cfg.aleph_alpha_embeddings.model_name)
    return client, tokenizer


client, tokenizer = setup_tokenizer_client()


def split_text(text: str):
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


# Settings for the text splitter
splitter = NLTKTextSplitter(length_function=count_tokens, chunk_size=300, chunk_overlap=50)


@load_config(location="config/db.yml")
def initialize_aleph_alpha_vector_db(cfg: DictConfig) -> None:
    """Initializes the Aleph Alpha vector db.

    Args:
    ----
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)

    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=5120, distance=models.Distance.COSINE),
        )
        logger.info(f"SUCCESS: Collection {collection_name} created.")


@load_config(location="config/db.yml")
def setup_connection_vector_db(cfg: DictConfig) -> Qdrant:
    """Sets up the connection to the vector db.

    Args:
    ----
        cfg (DictConfig): Configuration from the file

    Returns:
    -------
        Qdrant: The vector db
    """
    embedding = AlephAlphaAsymmetricSemanticEmbedding(
        model=cfg.aleph_alpha_embeddings.model_name, aleph_alpha_api_key=aleph_alpha_token, normalize=cfg.aleph_alpha_embeddings.normalize, compress_to_size=None
    )
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)

    vector_db = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def parse_txts(text: str, file_name: str, seperator: str, vector_db: Qdrant) -> None:
    """Parse the texts and add them to the vector db.

    Text should be marked with a link then </LINK> and then the text.

    Args:
    ----
        text (str): The text to parse
        file_name (str): Name of the file
        seperator (str): The seperator to split the text at
        vector_db (Qdrant): The vector db
    """
    # split every text in two parts one before </LINK> and one after
    link = text.split("</LINK>")[0]
    text = text.split("</LINK>")[1]

    # split the text at the seperator
    text_list: list = text.split(seperator)

    # check if first and last element are empty
    if not text_list[0]:
        text_list.pop(0)
    if not text_list[-1]:
        text_list.pop(-1)

    # meta data is a list of dicts including the "file_name" and the "link"
    metadata_list: list = []
    for _i in range(len(text_list)):
        metadata_list.append({"file_name": file_name, "link": link})

    vector_db.add_texts(texts=text_list, metadatas=metadata_list)


def parse_pdf(dir: str, vector_db: Qdrant) -> None:
    """Parse the pdfs and add them to the vector db.

    Args:
    ----
        dir (str): The directory to parse
        vector_db (Qdrant): The vector db
    """
    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFium2Loader)

    docs = loader.load_and_split(splitter)

    logger.info(f"Loaded {len(docs)} documents.")
    text_list = [doc.page_content for doc in docs]
    metadata_list = [doc.metadata for doc in docs]

    vector_db.add_texts(texts=text_list, metadatas=metadata_list)

    logger.info("SUCCESS: Texts added to Qdrant DB.")


def parse_json(dir: str, vector_db: Qdrant) -> None:
    """Parse the json and add them to the vector db.

    Args:
    ----
        dir (str): The directory to parse
        vector_db (Qdrant): The vector db
    """
    # open json file
    with open(dir) as file:
        json_file = json.load(file)

    token_list = []
    text_result_list = []
    metadata_list = []
    enriched_text_list = []
    for b in tqdm(json_file):
        metadata = json_file[b]["metadata"]
        identifier = b

        clean_text = json_file[b]["text"]

        # replace all linebreaks from teh text

        clean_text = re.sub(r"\s\s+", " ", clean_text)

        split_lenght = count_tokens(clean_text)
        # print(split_lenght)
        if split_lenght > 400:  # 400
            splits = split_text(clean_text)
            for s in tqdm(splits):
                # print(f"Number of tokens: {count_tokens(s)}")
                token_list.append(count_tokens(s))
                s = re.sub(r"\n", " ", s)
                # add to embedding list
                text_result_list.append(s)
                enriched_text_list.append(f"Author: {metadata['author']}, Title: {metadata['title']}, Chapter: {metadata['chapter']} Text: {s}")
                metadata_list.append(
                    {
                        "identifier": identifier,
                    }
                )

        else:
            clean_text = re.sub(r"\n", " ", clean_text)
            token_list.append(count_tokens(clean_text))
            # add to embedding list
            text_result_list.append(clean_text)
            enriched_text_list.append(f"Author: {metadata['author']}, Title: {metadata['title']}, Chapter: {metadata['chapter']} Text: {clean_text}")
            metadata_list.append(
                {
                    "identifier": identifier,
                }
            )

    # create a json object out of the lists
    # combine for one line in the json file the i element of the text result list the metadataa_lsit and the token_list
    final_split = {}
    for i in range(len(text_result_list)):
        final_split[i] = {"text": text_result_list[i], "metadata": metadata_list[i], "token": token_list[i], "clean_text": enriched_text_list[i]}

    # save the dict to a json file but as utf8
    with open("splits_nltk.json", "w", encoding="utf8") as outfile:
        json.dump(final_split, outfile, ensure_ascii=False)

        # start the embedding
    logger.info("Start the embedding!")
    vector_db.add_texts(texts=text_result_list, metadatas=metadata_list)
    logger.info("SUCCESS: Texts added to Qdrant DB.")


def main() -> None:
    """Main function to run the script."""
    load_dotenv()
    aleph_alpha_token = os.getenv("ALEPH_ALPHA_API_KEY")
    client, tokenizer = setup_tokenizer_client(aleph_alpha_token)
    NLTKTextSplitter(length_function=count_tokens, chunk_size=300, chunk_overlap=50)
    initialize_aleph_alpha_vector_db()
    vector_db = setup_connection_vector_db()

    parse_pdf(dir=Path("data/"), vector_db=vector_db)
    txt_dir = Path("data/txt")
    for file_name in txt_dir.iterdir():
        with file_name.open() as f:
            text = f.read()
            parse_txts(text=text, file_name=file_name.name, seperator="###", vector_db=vector_db)


if __name__ == "__main__":
    main()

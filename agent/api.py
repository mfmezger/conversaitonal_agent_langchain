"""FastAPI Backend for the Knowledge Agent."""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from langchain.docstore.document import Document as LangchainDocument
from loguru import logger
from qdrant_client import models
from qdrant_client.http.models.models import UpdateResult
from starlette.responses import JSONResponse

from agent.backend.aleph_alpha_service import (
    custom_completion_prompt_aleph_alpha,
    embedd_documents_aleph_alpha,
    embedd_text_aleph_alpha,
    embedd_text_files_aleph_alpha,
    explain_qa,
    process_documents_aleph_alpha,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
    summarize_text_aleph_alpha,
)
from agent.backend.gpt4all_service import (
    custom_completion_prompt_gpt4all,
    embedd_documents_gpt4all,
    embedd_text_gpt4all,
    qa_gpt4all,
    search_documents_gpt4all,
    summarize_text_gpt4all,
)
from agent.backend.open_ai_service import (
    embedd_documents_openai,
    search_documents_openai,
    send_custom_completion_openai,
)
from agent.data_model.request_data_model import (
    CustomPromptCompletion,
    EmbeddTextFilesRequest,
    EmbeddTextRequest,
    ExplainQARequest,
    LLMProvider,
    QARequest,
    SearchRequest,
)
from agent.data_model.response_data_model import (
    EmbeddingResponse,
    ExplainQAResponse,
    QAResponse,
    SearchResponse,
)
from agent.utils.utility import (
    combine_text_from_list,
    create_tmp_folder,
    validate_token,
)
from agent.utils.vdb import load_vec_db_conn

# add file logger for loguru
logger.add("logs/file_{time}.log", backtrace=False, diagnose=False)
logger.info("Startup.")


def my_schema() -> dict:
    """Used to generate the OpenAPI schema.

    Returns
    -------
        FastAPI: FastAPI App
    """
    openapi_schema = get_openapi(
        title="Conversational AI API",
        version="1.0",
        description="Chat with your Documents using Conversational AI by Aleph Alpha and OpenAI.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# initialize the Fast API Application.
app = FastAPI(debug=True)
app.openapi = my_schema

load_dotenv()

# load the token from the environment variables, is None if not set.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALEPH_ALPHA_API_KEY = os.environ.get("ALEPH_ALPHA_API_KEY")
logger.info("Loading REST API Finished.")


@app.get("/")
def read_root() -> str:
    """Returns the welcome message.

    Returns
    -------
        str: The welcome message.
    """
    return "Welcome to the Simple Aleph Alpha FastAPI Backend!"


def embedd_documents_wrapper(folder_name: str, llm_provider: LLMProvider, token: str | None = None, collection_name: str | None = None) -> None:
    """Call the right embedding function for the chosen backend.

    Args:
    ----
        folder_name (str): Name of the temporary folder.
        llm_backend (str, optional): LLM provider. Defaults to "openai".
        token (str, optional): Token for the LLM Provider of choice. Defaults to None.

    Raises:
    ------
        ValueError: If an invalid LLM Provider is set.
    """
    token = validate_token(token=token, llm_backend=llm_provider, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    if llm_provider == LLMProvider.ALEPH_ALPHA:
        # Embedd the documents with Aleph Alpha
        logger.debug("Embedding Documents with Aleph Alpha.")
        embedd_documents_aleph_alpha(dir=folder_name, aleph_alpha_token=token, collection_name=collection_name)
    elif llm_provider == LLMProvider.OPENAI:
        # Embedd the documents with OpenAI
        logger.debug("Embedding Documents with OpenAI.")
        embedd_documents_openai(dir=folder_name, open_ai_token=token)

    elif llm_provider == LLMProvider.GPT4ALL:
        embedd_documents_gpt4all(dir=folder_name)
    else:
        msg = "Please provide either 'aleph-alpha' or 'openai' as a parameter. Other backends are not implemented yet."
        raise ValueError(msg)


@app.post("/collection/create/{llm_provider}/{collection_name}")
def create_collection(llm_provider: LLMProvider, collection_name: str) -> JSONResponse:
    """Create a new collection in the vector database.

    Args:
    ----
        llm_provider (LLMProvider): Name of the LLM Provider
        collection_name (str): Name of the Collection
    """
    qdrant_client, _ = load_vec_db_conn()

    if llm_provider == LLMProvider.ALEPH_ALPHA:
        generate_collection_aleph_alpha(qdrant_client=qdrant_client, collection_name=collection_name, embeddings_size=5120)
    elif llm_provider == LLMProvider.OPENAI:
        generate_collection_openai(qdrant_client=qdrant_client, collection_name=collection_name)
    elif llm_provider == LLMProvider.GPT4ALL:
        generate_collection_gpt4all(qdrant_client=qdrant_client, collection_name=collection_name)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {llm_provider}")

    # return a success message
    return JSONResponse(content={"message": f"Collection {collection_name} created."})


@app.post("/embeddings/documents")
async def post_embedd_documents(
    files: list[UploadFile] = File(...), llm_provider: str = "aa", token: str | None = None, collection_name: str | None = None
) -> EmbeddingResponse:
    """Uploads multiple documents to the backend.

    Args:
    ----
        files (List[UploadFile], optional): Uploaded files. Defaults to File(...).

    Returns:
    -------
        JSONResponse: The response as JSON.
    """
    logger.info("Embedding Multiple Documents")
    llm_provider = LLMProvider.normalize(llm_provider)
    token = validate_token(token=token, llm_backend=llm_provider, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        if tmp_dir is None or not Path(tmp_dir).exists():
            msg = "Please provide a temporary folder to save the files."
            raise ValueError(msg)

        if file_name is None:
            msg = "Please provide a file to save."
            raise ValueError(msg)

        with open(Path(tmp_dir) / file_name, "wb") as f:
            f.write(await file.read())

    embedd_documents_wrapper(folder_name=tmp_dir, llm_provider=llm_provider, token=token, collection_name=collection_name)

    return EmbeddingResponse(status="success", files=file_names)


@app.post("/embeddings/text/")
async def embedd_text(request: EmbeddTextRequest) -> EmbeddingResponse:
    """Embeds text in the database.

    Args:
    ----
        request (EmbeddTextRequest): The request parameters.

    Raises:
    ------
        ValueError: If no token is provided or if no LLM provider is specified.

    Returns:
    -------
        JSONResponse: A response indicating that the text was received and saved, along with the name of the file it was saved to.
    """
    logger.info("Embedding Text")
    token = validate_token(token=request.llm_backend.token, llm_backend=request.llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    logger.info(f"Requested Backend is: {request.llm_backend}")
    if request.llm_backend.llm_provider in {"aleph-alpha", "aleph_alpha", "aa"}:
        # Embedd the documents with Aleph Alpha
        embedd_text_aleph_alpha(text=request.text, file_name=request.file_name, aleph_alpha_token=token, seperator=request.seperator)
        # return a success notificaton
        return JSONResponse(content={"message": "Text received and saved.", "filenames": request.file_name})
    elif request.llm_backend.llm_provider == "openai":
        # Embedd the documents with OpenAI
        # TODO: Implement
        msg = "Not implemented yet."
        raise ValueError(msg)
    elif request.llm_backend.llm_provider == "gpt4all":
        embedd_text_gpt4all(text=request.text, file_name=request.file_name, seperator=request.seperator)

    else:
        msg = "Please provide either 'aleph-alpha', 'gpt4all' or 'openai' as a parameter. Other backends are not implemented yet."
        raise ValueError(msg)

    return EmbeddingResponse(status="success", files=[request.file_name])


@app.post("/embeddings/texts/files")
async def embedd_text_files(request: EmbeddTextFilesRequest) -> EmbeddingResponse:
    """Embeds text files in the database.

    Args:
    ----
        request (EmbeddTextFilesRequest): The request parameters.

    Raises:
    ------
        ValueError: If a file does not have a valid name, if no temporary folder is provided, or if no token or LLM provider is specified.

    Returns:
    -------
        JSONResponse: A response indicating that the files were received and saved, along with the names of the files they were saved to.
    """
    logger.info("Embedding Text Files")
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in request.files:
        file_name = file.filename
        file_names.append(file_name)

        if file_name is None:
            msg = "File does not have a valid name."
            raise ValueError(msg)

        # Save the files to the temporary folder
        if tmp_dir is None or not Path(tmp_dir).exists():
            msg = "Please provide a temporary folder to save the files."
            raise ValueError(msg)

        with open(Path(tmp_dir) / file_name, "wb") as f:
            f.write(await file.read())

    token = validate_token(token=request.token, llm_backend=request.search.llm_backend.llm_provider, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    if request.search.llm_backend is None:
        msg = "Please provide a LLM Provider of choice."
        raise ValueError(msg)

    embedd_text_files_aleph_alpha(folder=tmp_dir, aleph_alpha_token=token, seperator=request.seperator)

    return EmbeddingResponse(status="success", files=file_names)


@app.post("/semantic/search")
def search(request: SearchRequest) -> list[SearchResponse]:
    """Searches for a query in the vector database.

    Args:
    ----
        request (SearchRequest): The search request.

    Raises:
    ------
        ValueError: If the LLM provider is not implemented yet.

    Returns:
    -------
        List[str]: A list of matching documents.
    """
    logger.info("Searching for Documents")
    request.llm_backend.token = validate_token(
        token=request.llm_backend.token, llm_backend=request.llm_backend.llm_provider, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY
    )

    if request.llm_backend.llm_provider is None:
        msg = "Please provide a LLM Provider of choice."
        raise ValueError(msg)

    DOCS = search_database(request)

    if not DOCS:
        logger.info("No Documents found.")
        return JSONResponse(content={"message": "No documents found."})

    logger.info(f"Found {len(DOCS)} documents.")

    response = []
    try:
        for d in DOCS:
            score = d[1]
            text = d[0].page_content
            page = d[0].metadata["page"]
            source = d[0].metadata["source"]
            response.append(SearchResponse(text=text, page=page, source=source, score=score))
    except Exception:
        for d in DOCS:
            score = d[1]
            text = d[0].page_content
            source = d[0].metadata["source"]
            response.append(SearchResponse(text=text, page=0, source=source, score=score))

    return response


@app.post("/qa")
def question_answer(request: QARequest) -> QAResponse:
    """Answer a question based on the documents in the database.

    Args:
    ----
        request (QARequest): The request parameters.

    Raises:
    ------
        ValueError: Error if no query or token is provided.

    Returns:
    -------
        Tuple: Answer, Prompt and Meta Data
    """
    logger.info("Answering Question")
    # if the query is not provided, raise an error
    if request.search.query is None:
        msg = "Please provide a Question."
        raise ValueError(msg)

    request.search.llm_backend.token = validate_token(
        token=request.search.llm_backend.token, llm_backend=request.search.llm_backend.llm_provider, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY
    )

    # if the history flag is activated and the history is not provided, raise an error
    if request.history and request.history_list is None:
        msg = "Please provide a HistoryList."
        raise ValueError(msg)

    # summarize the history
    if request.history:
        # combine the texts
        text = combine_text_from_list(request.history_list)
        if request.search.llm_backend == LLMProvider.ALEPH_ALPHA:
            # summarize the text
            summary = summarize_text_aleph_alpha(text=text, token=request.search.llm_backend.token)
            # combine the history and the query
            request.search.query = f"{summary}\n{request.search.query}"

        elif request.search.llm_backend == LLMProvider.OPENAI:
            pass

        elif request.search.llm_backend == LLMProvider.GPT4ALL:
            # summarize the text
            summary = summarize_text_gpt4all(text=text)
            # combine the history and the query
            request.search.query = f"{summary}\n{request.search.query}"
        else:
            msg = f"Unsupported LLM provider: {request.search.llm_backend.llm_provider}"
            raise ValueError(msg)

    documents = search_database(request.search)

    # call the qa function
    if request.search.llm_backend.llm_provider == LLMProvider.ALEPH_ALPHA:
        answer, prompt, meta_data = qa_aleph_alpha(query=request.search.query, documents=documents, aleph_alpha_token=request.search.llm_backend.token)
    elif request.search.llm_backend.llm_provider == LLMProvider.OPENAI:
        # TODO:
        msg = f"Unsupported LLM provider: {request.search.llm_backend.llm_provider}"
        raise ValueError(msg)
    elif request.search.llm_backend.llm_provider == LLMProvider.GPT4ALL:
        answer, prompt, meta_data = qa_gpt4all(documents=documents, query=request.search.query, summarization=request.summarization, language=request.language)
    else:
        msg = f"Unsupported LLM provider: {request.search.llm_backend.llm_provider}"
        raise ValueError(msg)

    return QAResponse(answer=answer, prompt=prompt, meta_data=meta_data)


@app.post("/explanation/explain-qa")
def explain_question_answer(explain_request: ExplainQARequest) -> ExplainQAResponse:
    # query: Optional[str] = None, llm_backend: str = "aa", token: Optional[str] = None, amount: int = 1, threshold: float = 0.0
    """Answer a question & explains it based on the documents in the database. This only works with Aleph Alpha.

    This uses the normal qa but combines it with the explain function.

    Args:
    ----
        query (str, optional): _description_. Defaults to None.
        aa_or_openai (str, optional): _description_. Defaults to "openai".
        token (str, optional): _description_. Defaults to None.
        amount (int, optional): _description_. Defaults to 1.

    Raises:
    ------
        ValueError: Error if no query or token is provided.

    Returns:
    -------
        Tuple: Answer, Prompt and Meta Data
    """
    logger.info("Answering Question and Explaining it.")
    # if the query is not provided, raise an error
    if explain_request.qa_request.search.query is None:
        msg = "Please provide a Question."
        raise ValueError(msg)

    explain_request.qa_request.search.llm_backend.token = validate_token(
        token=explain_request.qa_request.search.llm_backend.token,
        llm_backend=explain_request.qa_request.search.llm_backend.llm_provider,
        aleph_alpha_key=ALEPH_ALPHA_API_KEY,
        openai_key=OPENAI_API_KEY,
    )

    documents = search_database(explain_request.qa_request.search)

    # call the qa function
    explanation, score, text, answer, meta_data = explain_qa(
        query=explain_request.qa_request.search.query,
        explain_threshold=explain_request.explain_threshold,
        document=documents,
        aleph_alpha_token=explain_request.qa_request.search.llm_backend.token,
    )

    return ExplainQAResponse(explanation=explanation, score=score, text=text, answer=answer, meta_data=meta_data)


@app.post("/process_document")
async def process_document(files: list[UploadFile] = File(...), llm_backend: str = "aa", token: str | None = None, type: str = "invoice") -> None:
    """Process a document.

    Args:
    ----
        files (UploadFile): _description_
        llm_backend (str, optional): _description_. Defaults to "openai".
        token (Optional[str], optional): _description_. Defaults to None.
        type (str, optional): _description_. Defaults to "invoice".

    Returns:
    -------
        JSONResponse: _description_
    """
    logger.info("Processing Document")
    token = validate_token(token=token, llm_backend=llm_backend, aleph_alpha_key=ALEPH_ALPHA_API_KEY, openai_key=OPENAI_API_KEY)

    # Create a temporary folder to save the files
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        if tmp_dir is None or not Path(tmp_dir).exists():
            msg = "Please provide a temporary folder to save the files."
            raise ValueError(msg)

        if file_name is None:
            msg = "Please provide a file to save."
            raise ValueError(msg)

        with open(Path(tmp_dir) / file_name, "wb") as f:
            f.write(await file.read())

    process_documents_aleph_alpha(folder=tmp_dir, token=token, type=type)


def search_database(request: SearchRequest) -> list[tuple[LangchainDocument, float]]:
    """Searches the database for a query.

    Args:
    ----
        request (SearchRequest): The request parameters.

    Raises:
    ------
        ValueError: If the LLM provider is not implemented yet.

    Returns:
    -------
        JSON List of Documents consisting of the text, page, source and score.
    """
    logger.info("Searching for Documents")

    if request.llm_backend.llm_provider == LLMProvider.ALEPH_ALPHA:
        # Embedd the documents with Aleph Alpha
        documents = search_documents_aleph_alpha(
            aleph_alpha_token=request.llm_backend.token,
            query=request.query,
            amount=request.amount,
            threshold=request.filtering.threshold,
            collection_name=request.filtering.collection_name,
        )
    elif request.llm_backend.llm_provider == LLMProvider.OPENAI:
        documents = search_documents_openai(
            open_ai_token=request.llm_backend.token,
            query=request.query,
            amount=request.amount,
            threshold=request.filtering.threshold,
            collection_name=request.filtering.collection_name,
        )
    elif request.llm_backend.llm_provider == LLMProvider.GPT4ALL:
        documents = search_documents_gpt4all(
            query=request.query,
            amount=request.amount,
            threshold=request.filtering.threshold,
            collection_name=request.filtering.collection_name,
        )
    else:
        msg = f"Unsupported LLM provider: {request.llm_backend}"
        raise ValueError(msg)

    logger.info(f"Found {len(documents)} documents.")
    return documents


@app.post("/llm/completion/custom")
async def custom_prompt_llm(request: CustomPromptCompletion) -> str:
    """This method sents a custom completion request to the LLM Provider.

    Args:
    ----
        request (CustomPromptCompletion): The request parameters.

    Raises:
    ------
        ValueError: If the LLM provider is not implemented yet.
    """
    logger.info("Sending Custom Completion Request")
    if request.search.llm_backend == LLMProvider.ALEPH_ALPHA:
        # sent a completion
        answer = custom_completion_prompt_aleph_alpha(
            prompt=request.prompt,
            token=request.token,
            model=request.model,
            stop_sequences=request.stop_sequences,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    elif request.search.llm_backend == LLMProvider.OPENAI:
        answer = send_custom_completion_openai(
            prompt=request.prompt,
            token=request.token,
            model=request.model,
            stop_sequences=request.stop_sequences,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    elif request.search.llm_backend == LLMProvider.GPT4ALL:
        answer = custom_completion_prompt_gpt4all(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

    else:
        msg = f"Unsupported LLM provider: {request.search.llm_backend}"
        raise ValueError(msg)

    return answer


@app.delete("/embeddings/delete/{llm_provider}/{page}/{source}")
def delete(
    page: int,
    source: str,
    llm_provider: LLMProvider = LLMProvider.OPENAI,
) -> UpdateResult:
    """Delete a Vector from the database based on the page and source.

    Args:
    ----
        page (int): The page of the Document
        source (str): The name of the Document
        llm_provider (LLMProvider, optional): The LLM Provider. Defaults to LLMProvider.OPENAI.

    Returns:
    -------
        UpdateResult: The result of the Deletion Operation from the Vector Database.
    """
    logger.info("Deleting Vector from Database")
    if llm_provider == LLMProvider.ALEPH_ALPHA:
        collection = "aleph-alpha"
    elif llm_provider == LLMProvider.OPENAI:
        collection = "openai"
    elif llm_provider == LLMProvider.GPT4ALL:
        collection = "gpt4all"
    else:
        msg = f"Unsupported LLM provider: {llm_provider}"
        raise ValueError(msg)

    qdrant_client, _ = load_vec_db_conn()

    result = qdrant_client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.page",
                        match=models.MatchValue(value=page),
                    ),
                    models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source)),
                ],
            )
        ),
    )

    logger.info("Deleted Point from Database via Metadata.")
    return result


def initialize_aleph_alpha_vector_db() -> None:
    """Initializes the Aleph Alpha vector db.

    Args:
    ----
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = load_vec_db_conn()
    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_aa)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_aa} already exists.")
    except Exception:
        generate_collection_aleph_alpha(qdrant_client, collection_name=cfg.qdrant.collection_name_aa, embeddings_size=cfg.aleph_alpha_embeddings.size)


def generate_collection_aleph_alpha(qdrant_client, collection_name, embeddings_size) -> None:
    """Generate a collection for the Aleph Alpha Backend.

    Args:
    ----
        qdrant_client (_type_): _description_
        collection_name (_type_): _description_
        embeddings_size (_type_): _description_
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_open_ai_vector_db() -> None:
    """Initializes the OpenAI vector db.

    Args:
    ----
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = load_vec_db_conn()

    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_openai)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_openai} already exists.")
    except Exception:
        generate_collection_openai(qdrant_client, collection_name=cfg.qdrant.collection_name_openai)


def generate_collection_openai(qdrant_client: Qdrant, collection_name: str) -> None:
    """Generate a collection for the OpenAI Backend.

    Args:
    ----
        qdrant_client (_type_): Qdrant Client Langchain.
        collection_name (_type_): Name of the Collection
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_gpt4all_vector_db() -> None:
    """Initializes the GPT4ALL vector db.

    Args:
    ----
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = load_vec_db_conn()

    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_gpt4all)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_gpt4all} already exists.")
    except Exception:
        generate_collection_gpt4all(qdrant_client, collection_name=cfg.qdrant.collection_name_gpt4all)


def generate_collection_gpt4all(qdrant_client, collection_name: str) -> None:
    """Generate a collection for the GPT4ALL Backend.

    Args:
    ----
        qdrant_client (Qdrant): Qdrant Client
        collection_name (str): Name of the Collection
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


# initialize the databases
initialize_open_ai_vector_db()
initialize_aleph_alpha_vector_db()
initialize_gpt4all_vector_db()

# for debugging useful.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

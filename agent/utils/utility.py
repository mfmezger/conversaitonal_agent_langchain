"""This is the utility module."""
import uuid
from pathlib import Path

from langchain.prompts import PromptTemplate
from lingua import Language, LanguageDetectorBuilder
from loguru import logger

from agent.data_model.request_data_model import LLMProvider

# add new languages to detect here
languages = [Language.ENGLISH, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.7).build()


def combine_text_from_list(input_list: list) -> str:
    """Combines all strings in a list to one string.

    Args:
    ----
        input_list (list): List of strings

    Raises:
    ------
        TypeError: Input list must contain only strings

    Returns:
    -------
        str: Combined string
    """
    # iterate through list and combine all strings to one
    combined_text = ""

    logger.info(f"List: {input_list}")

    for text in input_list:
        # verify that text is a string
        if isinstance(text, str):
            # combine the text in a new line
            combined_text += "\n".join(text)

        else:
            msg = "Input list must contain only strings"
            raise TypeError(msg)

    return combined_text


def generate_prompt(prompt_name: str, text: str, query: str = "", language: str = "detect") -> str:
    """Generates a prompt for the Luminous API using a Jinja template.

    Args:
    ----
        prompt_name (str): The name of the file containing the Jinja template.
        text (str): The text to be inserted into the template.
        query (str): The query to be inserted into the template.
        language (str): The language the query should output. Or it can be detected

    Returns:
    -------
        str: The generated prompt.

    Raises:
    ------
        FileNotFoundError: If the specified prompt file cannot be found.
    """
    try:
        if language == "detect":
            detected_lang = detector.detect_language_of(query)
            if detected_lang == "Language.ENGLISH":
                language = "en"
            elif detected_lang == "Language.GERMAN":
                language = "de"
            else:
                logger.info(f"Detected Language is not supported. Using English. Detected language was {detected_lang}.")
                language = "en"

        if language not in {"en", "de"}:
            msg = "Language not supported."
            raise ValueError(msg)

        with open(Path("prompts", language, prompt_name), encoding="utf-8") as f:
            prompt = PromptTemplate.from_template(f.read(), template_format="jinja2")
    except FileNotFoundError:
        msg = f"Prompt file '{prompt_name}' not found."
        raise FileNotFoundError(msg)

    return prompt.format(text=text, query=query) if query else prompt.format(text=text)


def create_tmp_folder() -> str:
    """Creates a temporary folder for files to store.

    Returns
    -------
        str: The directory name.
    """
    # Create a temporary folder to save the files
    tmp_dir = Path.cwd() / f"tmp_{uuid.uuid4()}"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new folder {tmp_dir}.")
    except Exception as e:
        logger.error(f"Failed to create directory {tmp_dir}. Error: {e}")
        raise
    return str(tmp_dir)


def get_token(token: str | None, llm_provider: str | LLMProvider | None, aleph_alpha_key: str | None, openai_key: str | None) -> str:
    """Get the token from the environment variables or the parameter.

    Args:
    ----
        token (str, optional): Token from the REST service.
        llm_provider (Union[str, LLMProvider], optional): LLM provider. Defaults to "openai".

    Returns:
    -------
        str: Token for the LLM Provider of choice.

    Raises:
    ------
        ValueError: If no token is provided.
    """
    if isinstance(llm_provider, str):
        llm_provider = LLMProvider.normalize(llm_provider)

    if token in ("string", ""):
        token = None

    if token:
        return token

    env_token = aleph_alpha_key if llm_provider == LLMProvider.ALEPH_ALPHA else openai_key
    if not env_token and not token:
        msg = "No token provided."
        raise ValueError(msg)

    return env_token  # type: ignore


def validate_token(token: str | None, llm_backend: str | LLMProvider, aleph_alpha_key: str | None, openai_key: str | None) -> str:
    """Test if a token is available, and raise an error if it is missing when needed.

    Args:
    ----
        token (str): Token from the request
        llm_backend (str): Backend from the request
        aleph_alpha_key (str): Key from the .env file
        openai_key (str): Key from the .env file

    Raises:
    ------
        ValueError: If the llm backend is AA or OpenAI and there is no token.

    Returns:
    -------
        str: Token
    """
    return get_token(token, llm_backend, aleph_alpha_key, openai_key) if llm_backend != "gpt4all" else "gpt4all"


if __name__ == "__main__":
    # test the function
    generate_prompt("aleph_alpha_qa.j2", "This is a test text.", "What is the meaning of life?")

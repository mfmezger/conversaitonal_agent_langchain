"""The gui for the semantic search."""
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

url_search = "http://agent:8001/semantic/search"


logger.info("Starting Application.")

# Set small icon in the tab bar
st.set_page_config(page_title="Conversational Agent", page_icon=":rocket:", layout="wide")

# Create title
st.title("Semantic Search")


def create_folder_structure(folder_path: str) -> None:
    """Create the folder structure."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def initialize() -> None:
    """Initialize the GUI."""
    aleph_alpha_api_key = ""
    # The user needs to enter the aleph alpha api key

    st.session_state.api_key = aleph_alpha_api_key
    logger.debug("API Key was entered")

    col_ikey, col_number, query_or_abstract, col_llm_provider, col_collection_name = st.columns(5)
    with col_ikey:
        aleph_alpha_api_key = st.text_input("Aleph Alpha Token", type="password")
        st.session_state.api_key = aleph_alpha_api_key
    with col_number:
        amount = st.selectbox("Number of Documents", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=4)
    with query_or_abstract:
        # enable the choice between query and abstract
        query_or_abstract = st.radio("Query or Abstract", ["Query", "Abstract"])
    with col_llm_provider:
        llm_provider = st.radio("LLM Provider", ["aa", "gpt4all"])
    with col_collection_name:
        collection_name = st.text_input("Collection Name", value="")

    # create a select box to choose  the number of answers from 1-10
    # Search the documents
    search_query = st.text_area("Search Query", height=100)
    if st.button("Start Search", key="start_search") and search_query:
        logger.debug("Search was started")

        with st.spinner("Waiting for response...."):
            if collection_name == "text":
                collection_name = None
            documents = requests.post(
                url_search,
                json={
                    "query": search_query,
                    "llm_backend": llm_provider,
                    "token": st.session_state.api_key,
                    "collection_name": collection_name,
                    "amount": amount,
                },
            ).json()
            with st.chat_message(name="ai", avatar="🤖"):
                # Search the documents
                # make this one hidden
                # iterate over the objects in the json documents
                for d in documents:
                    with st.expander("Show Results", expanded=False):
                        st.write("_____")
                        col1, col2, col3 = st.columns(3)
                        col3.markdown(f"### Source: {d['source']}")
                        col1.markdown(f"### Page: {d['page']}")
                        col2.markdown(f"### Score: {d['score']}")

                        st.write(f"Text: {d['text']}")
                        st.write("_____")


# Start the GUI app
initialize()

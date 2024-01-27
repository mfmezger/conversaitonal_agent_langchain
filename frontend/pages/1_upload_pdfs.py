"""The page to upload  a pdf."""

import streamlit as st
from loguru import logger

from agent.backend.aleph_alpha_service import embedd_documents_aleph_alpha


def upload_files(save_path_input: str) -> list[tuple[str, bytes]]:
    """Upload PDF files and save them to the file system."""
    uploaded_files = st.file_uploader("Upload PDF Files", type=PDF_FILE_TYPE, accept_multiple_files=True)
    files = []

    for file in uploaded_files:
        with open(f"{save_path_input}{file.name}", "wb") as f:
            f.write(file.getbuffer())
        files.append((file.name, file.getbuffer()))
    return files


def start_embedding(file_path: str, token: str) -> None:
    """Start the embedding process."""
    embedd_documents_aleph_alpha(dir=file_path, aleph_alpha_token=token)


# Upload PDF files
files = upload_files(save_path_input)

# Start the embedding process
if st.button("Start Embedding", key="start_embedding"):
    logger.debug("Embedding was started")
    start_embedding(save_path_input, aleph_alpha_api_key)

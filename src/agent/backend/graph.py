"""Defines the graph structure and components for a conversational AI agent.

This module implements a RAG (Retrieval-Augmented Generation) system using LangChain
and LangGraph. It includes document retrieval, question answering, and dynamic
routing based on the conversation state and selected language model.
"""

import os
from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from agent.data_model.request_data_model import LLMProvider
from agent.utils.prompts import load_prompts
from agent.utils.utility import format_docs_for_citations
from agent.utils.vdb import load_vec_db_conn
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import ConfigurableField, RunnableConfig, chain
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from langgraph.graph import END, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph

# Constants for model keys
OPENAI_MODEL_KEY = "gpt-4o"
COHERE_MODEL_KEY = "cohere_command"
OLLAMA_MODEL_KEY = "ollama_llama8b3.1"

cohere_response_template, rephrase_template, response_template = load_prompts()


class AgentState(TypedDict):
    """Represents the state of the Agent during conversation."""

    query: str
    documents: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]


# Define language models
gpt4o = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
cohere_command = ChatCohere(
    model="command",
    temperature=0,
    cohere_api_key=os.environ.get("COHERE_API_KEY", "not_provided"),
    streaming=True,
)
ollama_chat = ChatOllama(model="llama3.1")

# Configure model alternatives with fallbacks
llm = gpt4o.configurable_alternatives(
    ConfigurableField(id="model_name"),
    default_key=LLMProvider.OPENAI.value,
    **{
        LLMProvider.COHERE.value: cohere_command,
    },
).with_fallbacks([cohere_command, ollama_chat])


def get_score_retriever() -> BaseRetriever:
    """Creates a retriever that includes similarity scores with retrieved documents.

    Returns
    -------
        BaseRetriever: Retriever with scoring capability.

    """
    embedding = CohereEmbeddings(model="embed-multilingual-v3.0")
    qdrant_client = load_vec_db_conn()
    vector_db = Qdrant(client=qdrant_client, collection_name="cohere", embeddings=embedding)

    @chain
    def retriever_with_score(query: str) -> list[Document]:
        """Defines a retriever that returns the score.

        Args:
        ----
            query (str): Query the user asks.

        Returns:
        -------
            list[Document]: List of Langchain Documents.

        """
        docs, scores = zip(*vector_db.similarity_search_with_score(query), strict=False)
        for doc, score in zip(docs, scores, strict=False):
            doc.metadata["score"] = score
        return docs

    return retriever_with_score


def get_retriever() -> BaseRetriever:
    """Creates a standard vector database retriever without scoring.

    Returns
    -------
        BaseRetriever: Qdrant retriever with Cohere embeddings.

    """
    embedding = CohereEmbeddings(model="embed-multilingual-v3.0")
    qdrant_client = load_vec_db_conn()
    vector_db = Qdrant(client=qdrant_client, collection_name="cohere", embeddings=embedding)
    return vector_db.as_retriever(search_kwargs={"k": 4})


def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieves relevant documents from the retriever based on the user's query.

    Args:
    ----
        state (AgentState): Current state of the agent.

    Returns:
    -------
        AgentState: Updated state with retrieved documents.

    """
    retriever = get_retriever()
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    relevant_documents = retriever.invoke(query)
    return {"query": query, "documents": relevant_documents}


def retrieve_documents_with_chat_history(state: AgentState) -> AgentState:
    """Retrieves relevant documents from the retriever based on the user's query and the chat history.

    Args:
    ----
        state (AgentState): Current state of the agent.

    Returns:
    -------
        AgentState: Updated state with retrieved documents.

    """
    retriever = get_retriever()
    model = llm.with_config(tags=["nostream"])

    condense_queston_prompt = PromptTemplate.from_template(rephrase_template)
    condense_question_chain = (condense_queston_prompt | model | StrOutputParser()).with_config(
        run_name="CondenseQuestion",
    )

    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    retriever_with_condensed_question = condense_question_chain | retriever
    relevant_documents = retriever_with_condensed_question.invoke({"question": query, "chat_history": get_chat_history(messages[:-1])})
    return {"query": query, "documents": relevant_documents}


def route_to_retriever(state: AgentState) -> Literal["retriever", "retriever_with_chat_history"]:
    """Determines which retriever to use based on the conversation state.

    Args:
    ----
        state (AgentState): Current state of the agent.

    Returns:
    -------
        Literal["retriever", "retriever_with_chat_history"]: Chosen retriever method.

    """
    return "retriever" if len(state["messages"]) == 1 else "retriever_with_chat_history"


def get_chat_history(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    """Extracts relevant chat history from a sequence of messages.

    Args:
    ----
        messages (Sequence[BaseMessage]): Full message history.

    Returns:
    -------
        list[dict[str, str]]: Formatted chat history for context.

    """
    return [
        {"content": message.content, "role": message.type}
        for message in messages
        if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(message, HumanMessage)
    ]


def generate_response(state: AgentState, model: LanguageModelLike, prompt_template: str) -> AgentState:
    """Generates a response using the provided language model and prompt template.

    Args:
    ----
        state (AgentState): Current state of the agent.
        model (LanguageModelLike): Language model to use for response generation.
        prompt_template (str): Template for the prompt.

    Returns:
    -------
        AgentState: Updated state with the generated response.

    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    response_synthesizer = prompt | model
    synthesized_response = response_synthesizer.invoke(
        {
            "question": state["query"],
            "context": format_docs_for_citations(state["documents"]),
            "chat_history": get_chat_history(convert_to_messages(state["messages"][:-1])),
        }
    )
    return {
        "messages": [synthesized_response],
    }


def generate_response_default(state: AgentState) -> AgentState:
    """Generates a response using the default language model and prompt template.

    Args:
    ----
        state (AgentState): Current state of the agent.

    Returns:
    -------
        AgentState: Updated state with the generated response.

    """
    return generate_response(state, llm, response_template)


def generate_response_cohere(state: AgentState) -> AgentState:
    """Generates a response using the Cohere language model and prompt template.

    Args:
    ----
        state (AgentState): Current state of the agent.

    Returns:
    -------
        AgentState: Updated state with the generated response.

    """
    model = llm.bind(documents=state["documents"])
    return generate_response(state, model, cohere_response_template)


def route_to_response_synthesizer(
    state: AgentState,  # noqa: ARG001
    config: RunnableConfig,
) -> Literal["response_synthesizer", "response_synthesizer_cohere"]:
    """Determines which response synthesizer to use based on the selected model.

    Args:
    ----
        state (AgentState): Current state of the agent (unused, but kept for consistency).
        config (RunnableConfig): Configuration containing model selection.

    Returns:
    -------
        Literal["response_synthesizer", "response_synthesizer_cohere"]: Chosen synthesizer method.

    """
    model_name = config.get("configurable", {}).get("model_name", OPENAI_MODEL_KEY)
    return "response_synthesizer_cohere" if model_name == COHERE_MODEL_KEY else "response_synthesizer"


def build_graph() -> CompiledStateGraph:
    """Constructs the conversation flow graph for the agent.

    Returns
    -------
        StateGraph: Compiled graph representing the agent's conversation logic.

    """
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("retriever_with_chat_history", retrieve_documents_with_chat_history)
    workflow.add_node("response_synthesizer", generate_response_default)
    workflow.add_node("response_synthesizer_cohere", generate_response_cohere)

    # Set entry point
    workflow.set_conditional_entry_point(route_to_retriever)

    # Connect nodes
    workflow.add_conditional_edges("retriever", route_to_response_synthesizer)
    workflow.add_conditional_edges("retriever_with_chat_history", route_to_response_synthesizer)

    # Set end points
    workflow.add_edge("response_synthesizer", END)
    workflow.add_edge("response_synthesizer_cohere", END)

    return workflow.compile()


# Example usage (commented out):
# graph = build_graph()
# answer = graph.invoke({
#     "messages": [
#         {"role": "human", "content": "Who is Luke Skywalker's father?"},
#         {"role": "assistant", "content": "Luke Skywalker's father is Anakin Skywalker."},
#         {"role": "human", "content": "And who is his mother?"}
#     ]
# })
# print(answer)

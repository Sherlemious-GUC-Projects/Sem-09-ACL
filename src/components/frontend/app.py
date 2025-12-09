from typing import List
from src.utils.types import ContextChunk


def mock_generate_response(user_query: str, context: List[ContextChunk]) -> str:
    """
    STUB: Mock implementation of LLM Generation.
    Use this for testing the pipeline without the real logic.
    """
    context_text = "\n".join([c.text for c in context])
    return (
        f"Based on the context: {context_text}, here is the answer to '{user_query}'."
    )


def generate_response(user_query: str, context: List[ContextChunk]) -> str:
    """
    TODO: Implement real LLM Generation logic here.
    1. Deduplicate context.
    2. Construct Prompt.
    3. Call LLM.
    """
    pass


def render_ui():
    """
    TODO: Implement Streamlit UI loop here.
    """
    pass

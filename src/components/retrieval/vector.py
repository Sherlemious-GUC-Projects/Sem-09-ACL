from typing import List
from src.utils.types import ContextChunk, RetrievalSource


def mock_query_graph_vector(raw_query: str, k: int = 5) -> List[ContextChunk]:
    """
    STUB: Mock implementation of Vector Retrieval.
    Use this for testing the pipeline without the real logic.
    """
    return [
        ContextChunk(
            id="review_456",
            text="Passenger review: The food was terrible on this flight.",
            score=0.85,
            source=RetrievalSource.VECTOR,
            metadata={"sentiment": "negative"},
        )
    ]


def query_graph_vector(raw_query: str, k: int = 5) -> List[ContextChunk]:
    """
    TODO: Implement real Vector Retrieval logic here.
    1. Generate embedding.
    2. KNN search in Neo4j.
    """
    pass

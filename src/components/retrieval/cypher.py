from typing import List
from src.utils.types import ProcessedQuery, ContextChunk, RetrievalSource


def mock_query_graph_cypher(processed_input: ProcessedQuery) -> List[ContextChunk]:
    """
    STUB: Mock implementation of Cypher Retrieval.
    Use this for testing the pipeline without the real logic.
    """
    airport = (
        processed_input.entities[0].value if processed_input.entities else "Unknown"
    )
    return [
        ContextChunk(
            id="flight_123",
            text=f"Found flight 123 from {airport}",
            score=1.0,
            source=RetrievalSource.CYPHER,
            metadata={"flight": "123"},
        )
    ]


def query_graph_cypher(processed_input: ProcessedQuery) -> List[ContextChunk]:
    """
    TODO: Implement real Cypher Retrieval logic here.
    1. Map processed_input.intent to Cypher Template.
    2. Inject entities.
    3. Run against Neo4j.
    """
    pass

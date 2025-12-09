from src.utils.types import ProcessedQuery, IntentType, Entity


def mock_process_user_query(raw_query: str) -> ProcessedQuery:
    """
    STUB: Mock implementation of Input Ingestion.
    Use this for testing the pipeline without the real logic.
    """
    return ProcessedQuery(
        original_text=raw_query,
        intent=IntentType.FLIGHT_SEARCH,
        entities=[Entity(entity_type="AIRPORT", value="ORD")],
    )


def process_user_query(raw_query: str) -> ProcessedQuery:
    """
    TODO: Implement real Input Ingestion logic here.
    1. Clean text.
    2. Classify Intent.
    3. Extract Entities.
    """
    pass

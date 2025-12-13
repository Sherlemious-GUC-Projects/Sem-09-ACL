### ~~~ GLobal Imports ~~~ ###
from typing import List, FrozenSet
import re

### ~~~ Local Imports ~~~ ###
from src.utils.types import Entity, ProcessedQuery
from src.components.ingestion.processor import (
    ReferenceData,
    clean_text,
    extract_metrics,
    determine_intent,
)


### ~~~ FUNCTIONS ~~~ ###
def extract_airports_regex(text: str, valid_airports: FrozenSet[str]) -> List[Entity]:
    """
    Finds 3-letter codes in text that exist in our valid airport set using Regex.
    Args:
        text: Normalized input text (upper case).
        valid_airports: Set of known valid IATA codes.
    Returns:
        List[Entity]: List of validated Airport entities.
    """
    # Regex logic: \b matches word boundaries, ensuring we don't match substrings of words
    candidates = re.findall(r"\b[A-Z]{3}\b", text)

    # Validation logic: Only return codes that actually exist in our DB
    found = [
        Entity(entity_type="AIRPORT", value=code)
        for code in candidates
        if code in valid_airports
    ]
    return found


def extract_aircraft_regex(text: str, valid_aircraft: FrozenSet[str]) -> List[Entity]:
    """
    Finds aircraft names using substring matching against the reference set.
    Args:
        text: Normalized input text.
        valid_aircraft: Set of known aircraft descriptions.
    Returns:
        List[Entity]: List of validated Aircraft entities.
    """
    found = []

    # Naive O(N) scan.
    # Why? Aircraft names are irregular (e.g. "B777-200" vs "ERJ-175") and hard to Regex reliably without a strict list.
    for model in valid_aircraft:
        if model.upper() in text:
            found.append(Entity(entity_type="AIRCRAFT", value=model))

    return found


def process_query_regex(raw_query: str, ref_data: ReferenceData) -> ProcessedQuery:
    """
    Main pipeline function for the Regex/Deterministic Approach.
    Args:
        raw_query: The user's original input string.
        ref_data: The immutable ground truth data.
    Returns:
        ProcessedQuery: The structured representation of the user's intent and entities.
    """
    cleaned = clean_text(raw_query)

    # Compose the specific extractors
    entities = (
        extract_airports_regex(cleaned, ref_data.airports)
        + extract_aircraft_regex(cleaned, ref_data.aircraft_models)
        + extract_metrics(raw_query)
    )

    intent = determine_intent(raw_query, entities)

    return ProcessedQuery(original_text=raw_query, intent=intent, entities=entities)

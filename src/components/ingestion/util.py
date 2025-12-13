### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import List, FrozenSet
from dataclasses import dataclass

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import IntentType, Entity


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass(frozen=True)
class ReferenceData:
    """
    Immutable container for ground truth data derived from the CSV.
    Used to pass reference sets (airports, aircraft) to pure functions.
    """

    airports: FrozenSet[str]
    aircraft_models: FrozenSet[str]


### ~~~ FUNCTIONS ~~~ ###
def clean_text(text: str) -> str:
    """
    Standardizes input text for processing.
    Args:
        text: Raw user input.
    Returns:
        str: Uppercase, stripped text.
    """
    return text.strip().upper()


def extract_metrics(text: str) -> List[Entity]:
    """
    Maps keywords to known metric types using simple string matching.
    Args:
        text: The raw or cleaned user query.
    Returns:
        List[Entity]: Found metric entities.
    """
    metrics = []
    # Using lowercase for case-insensitive keyword matching
    text_lower = text.lower()

    # Keyword mapping: "why" logic -> if user mentions food, they want food scores.
    if any(w in text_lower for w in ["food", "meal", "eat", "catering"]):
        metrics.append(Entity(entity_type="METRIC", value="food_satisfaction_score"))

    if any(w in text_lower for w in ["delay", "late", "time", "arrival"]):
        metrics.append(Entity(entity_type="METRIC", value="arrival_delay_minutes"))

    if any(w in text_lower for w in ["miles", "distance", "far"]):
        metrics.append(Entity(entity_type="METRIC", value="actual_flown_miles"))

    return metrics


def determine_intent(text: str, entities: List[Entity]) -> IntentType:
    """
    Rule-based intent classification based on text keywords and present entities.

    Args:
        text: The raw user query.
        entities: List of entities already extracted from the query.

    Returns:
        IntentType: The classified intent.
    """
    text_lower = text.lower()
    entity_types = {e.entity_type for e in entities}

    # Boolean flags for readability
    has_metrics = "METRIC" in entity_types
    has_airports = "AIRPORT" in entity_types

    # Priority 1: Statistical/Aggregate queries
    if "route" in text_lower or "busiest" in text_lower:
        return IntentType.ROUTE_STATS

    # Priority 2: Analysis/Reasoning queries
    if "why" in text_lower and ("late" in text_lower or "delay" in text_lower):
        return IntentType.DELAY_ANALYSIS

    if has_metrics and "food" in text_lower:
        return IntentType.SATISFACTION_METRICS

    # Priority 3: Search queries (requires specific entities usually)
    if has_airports and (
        "flight" in text_lower or "show" in text_lower or "find" in text_lower
    ):
        return IntentType.FLIGHT_SEARCH

    return IntentType.UNKNOWN

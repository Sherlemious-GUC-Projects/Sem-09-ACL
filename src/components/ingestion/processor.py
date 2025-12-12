### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import List, FrozenSet
from dataclasses import dataclass
import csv
import re

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import ProcessedQuery, IntentType, Entity, pretty_print


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass(frozen=True)
class ReferenceData:
    """Immutable container for ground truth data derived from the CSV."""

    airports: FrozenSet[str]
    aircraft_models: FrozenSet[str]


### ~~~ FUNCTIONS ~~~ ###
def load_reference_data(csv_path: str) -> ReferenceData:
    """
    Reads the CSV once and produces an immutable reference structure.
    This is the only place where I/O happens in this module.
    Args:
        csv_path (str): Path to the CSV data file.
    Returns:
        ReferenceData: Immutable container with sets of valid airports and aircraft models.
    """
    airports = set()
    aircraft = set()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("origin_station_code"):
                    airports.add(row["origin_station_code"].upper())
                if row.get("destination_station_code"):
                    airports.add(row["destination_station_code"].upper())
                if row.get("fleet_type_description"):
                    aircraft.add(row["fleet_type_description"])
    except FileNotFoundError:
        print(
            f"Warning: Data file not found at {csv_path}. Reference data will be empty."
        )

    return ReferenceData(
        airports=frozenset(airports), aircraft_models=frozenset(aircraft)
    )


def clean_text(text: str) -> str:
    """
    Standardizes input text for processing.
    Args:
        text (str): Raw user input.
    Returns:
        str: Cleaned and uppercased text.
    """
    return text.strip().upper()


def extract_airports(text: str, valid_airports: FrozenSet[str]) -> List[Entity]:
    """
    Finds 3-letter codes in text that exist in our valid airport set.
    Args:
        text (str): Cleaned user input.
        valid_airports (FrozenSet[str]): Set of valid airport codes.
    Returns:
        List[Entity]: List of found airport entities.
    """
    # Regex for 3-letter words
    candidates = re.findall(r"\b[A-Z]{3}\b", text)

    # Filter against ground truth
    found = [
        Entity(entity_type="AIRPORT", value=code)
        for code in candidates
        if code in valid_airports
    ]
    return found


def extract_aircraft(text: str, valid_aircraft: FrozenSet[str]) -> List[Entity]:
    """
    Finds aircraft names. This is trickier as they are longer strings.
    We check if any valid aircraft string appears in the text.
    Args:
        text (str): Cleaned user input.
        valid_aircraft (FrozenSet[str]): Set of valid aircraft model names.
    Returns:
        List[Entity]: List of found aircraft entities.
    """
    # Simple substring matching (naive but functional for limited sets)
    found = []
    for model in valid_aircraft:
        # Check if the model name (e.g. "B777-200") is in the text
        if model.upper() in text:
            found.append(Entity(entity_type="AIRCRAFT", value=model))
    return found


def extract_metrics(text: str) -> List[Entity]:
    """
    Maps keywords to known metric types.
    Args:
        text (str): Raw user input.
    Returns:
        List[Entity]: List of found metric entities.
    """
    metrics = []
    # Keyword mapping (can be expanded)
    text_lower = text.lower()

    if any(w in text_lower for w in ["food", "meal", "eat", "catering"]):
        metrics.append(Entity(entity_type="METRIC", value="food_satisfaction_score"))

    if any(w in text_lower for w in ["delay", "late", "time", "arrival"]):
        metrics.append(Entity(entity_type="METRIC", value="arrival_delay_minutes"))

    if any(w in text_lower for w in ["miles", "distance", "far"]):
        metrics.append(Entity(entity_type="METRIC", value="actual_flown_miles"))

    return metrics


def extract_entities(text: str, ref_data: ReferenceData) -> List[Entity]:
    """
    Composes specific extractors.
    Args:
        text (str): Raw user input.
        ref_data (ReferenceData): Ground truth data for extraction.
    Returns:
        List[Entity]: Combined list of all extracted entities.
    """
    cleaned = clean_text(text)

    return (
        extract_airports(cleaned, ref_data.airports)
        + extract_aircraft(cleaned, ref_data.aircraft_models)
        + extract_metrics(text)
    )


def determine_intent(text: str, entities: List[Entity]) -> IntentType:
    """
    Rule-based intent classification based on text keywords and present entities.
    Functional logic: Inputs -> Output.
    Args:
        text (str): Raw user input.
        entities (List[Entity]): Extracted entities from the input.
    Returns:
        IntentType: Classified intent of the user query.
    """
    text_lower = text.lower()
    entity_types = {e.entity_type for e in entities}

    # Logic Rules
    has_metrics = "METRIC" in entity_types
    has_airports = "AIRPORT" in entity_types

    if "route" in text_lower or "busiest" in text_lower:
        return IntentType.ROUTE_STATS

    if "why" in text_lower and ("late" in text_lower or "delay" in text_lower):
        return IntentType.DELAY_ANALYSIS

    if has_metrics and "food" in text_lower:
        return IntentType.SATISFACTION_METRICS

    if has_airports and (
        "flight" in text_lower or "show" in text_lower or "find" in text_lower
    ):
        return IntentType.FLIGHT_SEARCH

    # Default fallback
    return IntentType.UNKNOWN


def process_user_query(raw_query: str, ref_data: ReferenceData) -> ProcessedQuery:
    """
    Main pipeline function.
    Args:
        raw_query (str): Raw user input.
        ref_data (ReferenceData): Ground truth data for extraction.
    Returns:
        ProcessedQuery: Structured representation of the user query.
    """
    # 1. Extraction
    entities = extract_entities(raw_query, ref_data)

    # 2. Intent Classification
    intent = determine_intent(raw_query, entities)

    return ProcessedQuery(original_text=raw_query, intent=intent, entities=entities)


def main() -> int:
    """"""
    raw_query = "Show me flights from JFK to LAX on a B777-200 and their delay stats"
    ref_data = load_reference_data("dbs/Airline_surveys_sample.csv")
    processed = process_user_query(raw_query, ref_data)
    pretty_print(processed)
    return 0


if __name__ == "__main__":
    main()

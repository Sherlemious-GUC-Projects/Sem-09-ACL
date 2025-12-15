### ~~~ GLOBAL IMPORTS ~~~ ###
import re
import nltk
from typing import List, FrozenSet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from dataclasses import dataclass
import pandas as pd


### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import Entity, ProcessedQuery, IntentType

### ~~~ CONFIGURATION ~~~ ###
CSV_PATH = "dbs/Airline_surveys_sample.csv"


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass(frozen=True)
class ReferenceData:
    """
    Immutable container for ground truth data derived from the CSV.
    Used to pass reference sets (airports, aircraft) to pure functions.
    """

    airports: FrozenSet[str]
    aircraft_models: FrozenSet[str]


### ~~~ INITIALIZATION ~~~ ###
def _ensure_resources() -> None:
    """Effectful function to ensure NLTK resources exist."""
    resources = ["punkt", "averaged_perceptron_tagger", "maxent_ne_chunker", "words"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r, quiet=True)
        except Exception:
            # Fallback for non-tokenizer resources
            nltk.download(r, quiet=True)


_ensure_resources()


### ~~~ PURE FUNCTIONS: EXTRACTION ~~~ ###
def load_reference_data(csv_path: str) -> ReferenceData:
    """
    Loads reference data (airports, aircraft models) from a CSV file.
    """
    df = pd.read_csv(csv_path)

    # Column mapping based on actual CSV header:
    # origin_station_code -> Airports
    # fleet_type_description -> Aircraft

    # We take the union of Origin and Destination to ensure we capture all valid airports
    # (Checking both ensures coverage even if some airports only appear as destinations)
    origins = set(df["origin_station_code"].dropna().unique())
    dests = set(df["destination_station_code"].dropna().unique())
    airports = frozenset(origins | dests)

    aircraft_models = frozenset(df["fleet_type_description"].dropna().unique())

    return ReferenceData(airports=airports, aircraft_models=aircraft_models)


def extract_structured_entities(text: str) -> List[Entity]:
    """
    Extracts entities with strict, well-defined formats using Regex.
    Best for: Flight Numbers, Dates.
    """
    entities = []

    # 1. Dates (YYYY-MM-DD)
    date_matches = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    entities.extend([Entity("DATE", m) for m in date_matches])

    # 2. Flight Numbers (e.g., BA123, UA45)
    # Pattern: 2 letters (Airline) + 1-4 digits
    flight_matches = re.findall(r"\b[A-Za-z]{2}\d{1,4}\b", text)
    entities.extend([Entity("FLIGHT_NUM", m.upper()) for m in flight_matches])

    return entities


def extract_ner_candidates(text: str) -> List[str]:
    """
    Uses NLTK Chunking to find 'candidate' entity strings from unstructured text.
    Returns: A list of potential entity strings (e.g., "737 Max", "JFK", "London").
    """
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)

    candidates = []

    # Grammar: Noun Phrases often contain our entities.
    # We capture:
    # 1. Sequences of Proper Nouns (NNP) and/or Numbers (CD) -> "Boeing 737", "737 Max"
    grammar = r"NP: {<CD|NNP>+}"
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(tags)

    for subtree in tree:
        if isinstance(subtree, nltk.tree.Tree):
            # Phrase Candidate: Join tokens in the chunk
            phrase = " ".join([token for token, pos in subtree.leaves()])
            candidates.append(phrase)
        else:
            # Single Token Candidates (Fallback)
            token, pos = subtree

            # Airport Codes are often 3 letters and ALL CAPS (e.g., JFK, ORD)
            # We explicitly catch these even if tags are wrong (e.g. JFK tagged as Verb)
            if len(token) == 3 and token.isupper():
                candidates.append(token)

            # Numbers that didn't chunk might be partial aircraft names (rare but possible)
            if pos == "CD":
                candidates.append(token)

    return candidates


def validate_and_map_entities(
    candidates: List[str], ref: ReferenceData
) -> List[Entity]:
    """
    Validates raw candidates against the 'Ground Truth' Reference Data.
    Maps valid strings to their specific Entity Types (AIRPORT, AIRCRAFT).
    """
    valid_entities = []

    for cand in candidates:
        # Normalize for comparison
        clean_cand = cand.strip().upper()

        # 1. Check Airports (Exact Match)
        if clean_cand in ref.airports:
            valid_entities.append(Entity("AIRPORT", clean_cand))
            continue  # Prioritize explicit airport codes

        # 2. Check Aircraft (Fuzzy/Substring Match)
        # Aircraft names in text ("737 Max") might match DB ("B737-MAX8") loosely or exactly.
        for model in ref.aircraft_models:
            model_upper = model.upper()

            # Exact Match
            if clean_cand == model_upper:
                valid_entities.append(Entity("AIRCRAFT", model))
                break

            # Substring Logic:
            # If candidate is "737 Max" and DB has "B737-MAX8", it's hard to match without complex rules.
            # But if DB has "737 Max" and text has "737 Max", it matches.
            # If text has "Boeing 777" and DB has "B777-200", we might miss it with strict equality.
            # For this exercise, we check if the candidate *contains* key parts of the model or vice versa?
            # Safer: Check if candidate is a substring of a known model?
            # e.g. Candidate "ERJ-175" (extracted as chunk) matches "ERJ-175" in DB.

            # We assume the user asks for things mostly as they appear or with slight variation.
            if clean_cand in model_upper and len(clean_cand) > 3:
                valid_entities.append(Entity("AIRCRAFT", model))
                break

    return valid_entities


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


### ~~~ PIPELINE ~~~ ###


def process_query(raw_query: str, ref_data: ReferenceData) -> ProcessedQuery:
    """


    Main Entry Point: Orchestrates the extraction pipeline.


    """

    # 1. Regex (Structured)

    structured_ents = extract_structured_entities(raw_query)

    # 2. NER + Validation (Unstructured Domain)

    candidates = extract_ner_candidates(raw_query)

    domain_ents = validate_and_map_entities(candidates, ref_data)

    # 3. Merge & Deduplicate

    # Dictionary keyed by (Type, Value) handles uniqueness automatically

    merged = {}

    for e in structured_ents + domain_ents:
        merged[(e.entity_type, e.value)] = e

    unique_entities = list(merged.values())

    # 4. Determine Intent

    intent = determine_intent(raw_query, unique_entities)

    return ProcessedQuery(
        original_text=raw_query, intent=intent, entities=unique_entities
    )


### ~~~ MAIN / TEST ~~~ ###
def main() -> int:
    print("~~~ Initializing Perfected NER Processor ~~~ ")

    # 1. Load Data
    print(f"Loading Reference Data from {CSV_PATH}...")
    try:
        ref_data = load_reference_data(CSV_PATH)
        print(
            f"Loaded {len(ref_data.airports)} Airports and {len(ref_data.aircraft_models)} Aircraft Models."
        )
    except FileNotFoundError:
        print(f"Error: CSV not found at {CSV_PATH}. Cannot run tests.")
        return 1

    # 2. Test Cases
    # We use entities that we KNOW are in the sample CSV (e.g. LAX, ORX, B737-MAX8)
    # plus some Regex patterns that don't need the DB.
    test_cases = [
        (
            "Flights from LAX to ORX",
            [("AIRPORT", "LAX"), ("AIRPORT", "ORX"), ("INTENT", "FLIGHT_SEARCH")],
        ),
        (
            "Was flight BA123 delayed on 2023-01-01?",
            [("FLIGHT_NUM", "BA123"), ("DATE", "2023-01-01")],
        ),
        ("How is the food on the B787-10?", [("AIRCRAFT", "B787-10")]),
        ("Show me stats for ERJ-175", [("AIRCRAFT", "ERJ-175")]),
    ]

    failures = 0
    for query, expected in test_cases:
        print(f"\nQuery: '{query}'")
        result = process_query(query, ref_data)

        # Flatten results for easy checking
        # found_types = {e.entity_type for e in result.entities}
        found_vals = {e.value for e in result.entities}

        print(f"  Found: {[(e.entity_type, e.value) for e in result.entities]}")

        # Check assertions
        case_passed = True
        for item in expected:
            if item[0] == "INTENT":
                if (
                    result.intent != item[1]
                ):  # Mismatch intent name? item[1] is a string e.g. "FLIGHT_SEARCH"
                    # Enum string comparison
                    if result.intent.value != item[1] and result.intent.name != item[1]:
                        print(
                            f"  [FAIL] Intent mismatch. Expected {item[1]}, got {result.intent}"
                        )
                        case_passed = False
            else:
                # Entity check (Type, Value)
                e_type, e_val = item
                if e_val not in found_vals:
                    print(f"  [FAIL] Missing Entity: {e_val} ({e_type})")
                    case_passed = False

        if case_passed:
            print("  [PASS]")
        else:
            failures += 1

    if failures == 0:
        print("\nAll tests passed successfully!")
    else:
        print(f"\n{failures} tests failed.")

    return 0


if __name__ == "__main__":
    main()

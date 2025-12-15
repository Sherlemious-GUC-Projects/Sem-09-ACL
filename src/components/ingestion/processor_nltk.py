### ~~~ GLOBAL IMPORTS ~~~ ###
import nltk
import re
from typing import List, FrozenSet, Optional
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import Entity, ProcessedQuery
from src.components.ingestion.util import (
    ReferenceData,
    extract_metrics,
    determine_intent,
)


### ~~~ STATE DEFINITIONS ~~~ ###
def _download_nltk_data():
    """
    Ensures all necessary NLTK models are available locally.
    We explicitly list all dependencies for the NER pipeline.
    """
    required_packages = [
        "punkt",  # Tokenizer
        "averaged_perceptron_tagger",  # POS Tagger (generic)
        "maxent_ne_chunker",  # NE Chunker Base
        "words",  # Word list for valid English words
    ]

    for package in required_packages:
        nltk.download(package, quiet=True, raise_on_error=False)


_download_nltk_data()


### ~~~ FUNCTIONS ~~~ ###
def _extract_flight_code(token: str) -> Optional[str]:
    """
    Helper to identify flight codes (e.g., 'BA123', 'AA45') using Regex.
    Standard pattern: 2 letters followed by 1-4 digits.
    """
    # Regex: Start of string, 2 uppercase letters, 1-4 digits, End of string
    # We use ignoring case for input, but pattern expects standard format
    match = re.fullmatch(r"([a-zA-Z]{2})\d{1,4}", token)
    if match:
        return token.upper()
    return None


def _extract_date(token: str) -> Optional[str]:
    """
    Helper to identify simple date formats.
    Supports: YYYY-MM-DD
    """
    # Very basic ISO date regex
    match = re.fullmatch(r"\d{4}-\d{2}-\d{2}", token)
    if match:
        return token
    return None


def extract_entities_nltk(
    text: str, valid_airports: FrozenSet[str], valid_aircraft: FrozenSet[str]
) -> List[Entity]:
    """
    Extracts entities using NLTK POS Tagging and a custom Chunking Grammar.
    This is more robust for domain-specific entities (like '737 Max') than
    the generic 'ne_chunk' model.

    Args:
        text: Raw user text.
        valid_airports: Reference set for validation.
        valid_aircraft: Reference set for validation.

    Returns:
        List[Entity]: List of discovered entities.
    """
    entities = []

    ### 1. Tokenize & POS Tag ###
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    ### 2. Custom Chunking Grammar ###
    # We define patterns for Noun Phrases that might be our entities.
    # NP: Optional Number (CD) followed by one or more Proper Nouns (NNP)
    #     Matches: "737 Max" (CD NNP), "JFK" (NNP), "Boeing 777" (NNP CD)
    grammar = r"""
      ENTITY_PHRASE: {<CD|NNP>+}   # Capture sequences of proper nouns and numbers
    """
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(tagged)

    ### 3. Traverse Tree ###
    for subtree in tree:
        if isinstance(subtree, nltk.tree.Tree):
            # We found a candidate phrase (e.g., "737 Max", "JFK")
            # Reconstruct the text for this chunk
            chunk_tokens = [token for token, tag in subtree.leaves()]
            chunk_text = " ".join(chunk_tokens)

            # Check 1: Is this whole chunk an Airport? (e.g. "JFK")
            clean_val = chunk_text.strip().upper()
            if clean_val in valid_airports:
                entities.append(Entity(entity_type="AIRPORT", value=clean_val))

            # Check 2: Is this chunk an Aircraft? (e.g. "737 Max")
            # We check if the chunk text *contains* a known model (relaxed match)
            # or matches exactly.
            for model in valid_aircraft:
                # distinct check to avoid partial matches inside words
                if model.upper() == chunk_text.upper():
                    entities.append(Entity(entity_type="AIRCRAFT", value=model))
                elif model.upper() in chunk_text.upper():
                    # Only accept partial if it covers significant tokens
                    # (Simple substring check here is usually safe for "737 Max")
                    entities.append(Entity(entity_type="AIRCRAFT", value=model))

            # Check 3: Flight Numbers in chunks (e.g. "BA 123" space separated)
            # If the chunk is "BA 123", regex on the joined string might catch it
            flight_match = _extract_flight_code(chunk_text.replace(" ", ""))
            if flight_match:
                entities.append(Entity(entity_type="FLIGHT_NUM", value=flight_match))

        else:
            # It's a single token that didn't fit the grammar (e.g. verbs, simple nouns)
            token_text, tag = subtree

            # Logic: Flight Numbers
            flight_code = _extract_flight_code(token_text)
            if flight_code:
                entities.append(Entity(entity_type="FLIGHT_NUM", value=flight_code))

            # Logic: Dates
            date_val = _extract_date(token_text)
            if date_val:
                entities.append(Entity(entity_type="DATE", value=date_val))

            # Logic: Airports (Robust Fallback for bad tags)
            # Catch 3-letter uppercase codes (e.g. JFK tagged as VB, ORD tagged as NN)
            # We enforce UPPERCASE matching to avoid common words like "sat", "sun", "mad"
            if (
                len(token_text) == 3
                and token_text.isupper()
                and token_text in valid_airports
            ):
                entities.append(Entity(entity_type="AIRPORT", value=token_text))

    return entities


def process_query_nltk(raw_query: str, ref_data: ReferenceData) -> ProcessedQuery:
    """
    Main pipeline function for the NLTK approach.
    """
    ### 1. NLTK/Regex Entity Extraction ###
    nltk_entities = extract_entities_nltk(
        raw_query, ref_data.airports, ref_data.aircraft_models
    )

    ### 2. Metric Extraction (Keyword based) ###
    metric_entities = extract_metrics(raw_query)

    ### 3. Merge & Deduplicate ###
    all_entities = nltk_entities + metric_entities

    unique_entities = []
    seen = set()
    for e in all_entities:
        key = (e.entity_type, e.value)
        if key not in seen:
            unique_entities.append(e)
            seen.add(key)

    ### 4. Intent Determination ###
    intent = determine_intent(raw_query, unique_entities)

    return ProcessedQuery(
        original_text=raw_query, intent=intent, entities=unique_entities
    )


def main() -> int:
    """
    Test suite for the NLTK processor.
    """
    print("Initializing NLTK Processor Test...")

    # Mock Reference Data
    mock_ref = ReferenceData(
        airports=frozenset(["LHR", "JFK", "ORD", "LAX", "DXB"]),
        aircraft_models=frozenset(["737 Max", "A380", "Dreamliner", "777"]),
    )

    test_cases = [
        (
            "Show me food ratings for flight BA123 from LHR to JFK",
            [
                "METRIC:food_satisfaction_score",
                "FLIGHT_NUM:BA123",
                "AIRPORT:LHR",
                "AIRPORT:JFK",
            ],
        ),
        (
            "Was the 737 Max delayed on 2023-01-01?",
            ["AIRCRAFT:737 Max", "METRIC:arrival_delay_minutes", "DATE:2023-01-01"],
        ),
        (
            "How is the service on the A380?",
            ["AIRCRAFT:A380"],
            # Note: 'service' might not map to a metric in default extract_metrics unless 'food'/'delay' etc are present,
            # checking util.py extract_metrics: 'food', 'meal', 'eat', 'catering' -> food. 'delay' -> delay.
            # 'service' is not in the default list found in util.py. So we expect only AIRCRAFT.
        ),
        ("Flights leaving ORD", ["AIRPORT:ORD"]),
    ]

    failures = 0
    for query, expected_strings in test_cases:
        print(f"\nProcessing: '{query}'")
        result = process_query_nltk(query, mock_ref)

        # Convert result entities to strings for easy comparison
        found_strings = [f"{e.entity_type}:{e.value}" for e in result.entities]

        # Check if all expected strings are found
        # (We use set logic for loose ordering checks)
        missing = [ex for ex in expected_strings if ex not in found_strings]

        if missing:
            print(f"  [FAILED] Missing entities: {missing}")
            print(f"  Found: {found_strings}")
            failures += 1
        else:
            print(f"  [PASSED] Found: {found_strings}")
            print(f"  Intent: {result.intent}")

    if failures == 0:
        print("\nAll tests passed successfully!")
    else:
        print(f"\n{failures} tests failed.")

    return 0


if __name__ == "__main__":
    main()

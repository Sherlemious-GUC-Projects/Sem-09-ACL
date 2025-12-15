### ~~~ GLOBAL IMPORTS ~~~ ###
from nltk.tokenize import word_tokenize
from typing import List, FrozenSet
from dataclasses import dataclass
from nltk.tag import pos_tag
import pandas as pd
import nltk
import re


### ~~~ LOCAL IMPORTS ~~~ ###
from utils.types import Entity, IntentType, EntityType
from analyise_sentiment import classifier


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
    entities.extend([Entity(EntityType.DATE, m) for m in date_matches])

    # 2. Flight Numbers (e.g., BA123, UA45)
    # Pattern: 2 letters (Airline) + 1-4 digits
    flight_matches = re.findall(r"\b[A-Za-z]{2}\d{1,4}\b", text)
    entities.extend(
        [Entity(EntityType.FLIGHT_NUMBER, m.upper()) for m in flight_matches]
    )

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
    # Updated to capture Adjectives (next) and general Nouns (winter)
    # <JJ>? : Optional Adjective (e.g. "next", "past")
    # <NN.*|CD>+ : One or more Nouns (proper or common) or Numbers
    grammar = r"NP: {<JJ>?<NN.*|CD>+}"
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
            if len(token) == 3 and token.isupper():
                candidates.append(token)

    return candidates


def clean_entity_value(text: str) -> str:
    """
    Removes leading/trailing lowercase words from a string.
    Useful for cleaning candidates like "new A321neo configuration" -> "A321neo".
    """
    tokens = text.split()
    if not tokens:
        return text

    # Remove from start
    while tokens and tokens[0].islower() and not tokens[0].isdigit():
        tokens.pop(0)

    # Remove from end
    while tokens and tokens[-1].islower() and not tokens[-1].isdigit():
        tokens.pop()

    if not tokens:
        return text  # Return original if everything was stripped (safety)

    return " ".join(tokens)


def validate_and_map_entities(
    candidates: List[str], ref: ReferenceData
) -> List[Entity]:
    """
    Validates raw candidates using a Hybrid approach:
    1. Ground Truth Check (CSV)
    2. Regex Heuristics (IATA codes)
    3. Zero-Shot Classification (ML) for OOV entities
    """
    valid_entities = []

    # Allowed lowercase temporal words (whitelist)
    valid_temporal = {
        "winter",
        "summer",
        "spring",
        "fall",
        "year",
        "month",
        "week",
        "day",
    }

    # Candidate Labels for the Zero-Shot Model
    labels = ["airport code", "aircraft model", "date", "city", "general text"]

    for cand in candidates:
        # Clean: remove trailing 's (A380's -> A380)
        clean_cand = cand.strip()
        if clean_cand.lower().endswith("'s"):
            clean_cand = clean_cand[:-2]

        cand_upper = clean_cand.upper()

        # 1. Exact Match Check (Reference Data)
        if cand_upper in ref.airports:
            valid_entities.append(Entity(EntityType.AIRPORT, cand_upper))
            continue

        for model in ref.aircraft_models:
            if cand_upper == model.upper() or (
                len(cand_upper) > 3 and cand_upper in model.upper()
            ):
                valid_entities.append(Entity(EntityType.AIRCRAFT, model))
                break
        else:
            # 2. Heuristic: 3-Letter Uppercase Code -> Likely Airport (IATA)
            # (Solves LHR, ORD, CDG missing from CSV)
            if re.fullmatch(r"[A-Z]{3}", clean_cand):
                valid_entities.append(Entity(EntityType.AIRPORT, clean_cand))
                continue

            # 3. Heuristic: Noise Filter
            # Drop lowercase words unless they contain whitelisted temporal terms
            is_title_or_upper = clean_cand[0].isupper() or any(
                c.isupper() for c in clean_cand
            )
            is_digit = any(c.isdigit() for c in clean_cand)

            # Check if any token in the candidate is a valid temporal word
            cand_lower_tokens = set(clean_cand.lower().split())
            is_valid_temporal = not cand_lower_tokens.isdisjoint(valid_temporal)

            if not (is_title_or_upper or is_digit or is_valid_temporal):
                continue

            # 4. Zero-Shot Classification (ML)
            try:
                result = classifier(clean_cand, candidate_labels=labels)
                top_label = result["labels"][0]  # type: ignore
                score = result["scores"][0]  # type: ignore

                if score > 0.6:
                    if top_label == "airport code":
                        # Clean to remove noise if any
                        cleaned = clean_entity_value(clean_cand)
                        valid_entities.append(Entity(EntityType.AIRPORT, cleaned))
                    elif top_label == "aircraft model":
                        # Clean "new A321neo configuration" -> "A321neo"
                        cleaned = clean_entity_value(clean_cand)
                        valid_entities.append(Entity(EntityType.AIRCRAFT, cleaned))
                    elif top_label == "date":
                        # Do NOT clean dates (we need "past year")
                        valid_entities.append(Entity(EntityType.DATE, clean_cand))
                    # 'city' and 'general text' are intentionally dropped
            except Exception:
                pass

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
    has_metrics = EntityType.METRIC in entity_types
    has_airports = EntityType.AIRPORT in entity_types

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


def extract_ner(raw_query: str, ref_data: ReferenceData) -> list[Entity]:
    """
    Full NER Processing Pipeline:
    1. Structured Extraction via Regex
    2. Unstructured Extraction via NER + Validation
    3. Merging & Deduplication
    Args:
        raw_query: The user's raw text query.
        ref_data: ReferenceData containing ground truth sets.
    Returns:
        List[Entity]: The final list of unique extracted entities.
    """
    # 1. Regex (Structured)
    structured_ents: list[Entity] = extract_structured_entities(raw_query)

    # 2. NER + Validation (Unstructured Domain)
    candidates: list[str] = extract_ner_candidates(raw_query)
    domain_ents: list[Entity] = validate_and_map_entities(candidates, ref_data)

    # 3. Merge & Deduplicate
    # Dictionary keyed by (Type, Value) handles uniqueness automatically
    merged: dict[tuple[str, str], Entity] = {}
    for e in structured_ents + domain_ents:
        merged[(e.entity_type, e.value)] = e
    unique_entities: list[Entity] = list(merged.values())

    return unique_entities


if __name__ == "__main__":
    pass

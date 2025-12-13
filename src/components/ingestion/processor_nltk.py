### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import List, FrozenSet, Set, Tuple
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import nltk
import re

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import Entity, ProcessedQuery
from src.components.ingestion.processor import (
    ReferenceData,
    extract_metrics,
    determine_intent,
)


### ~~~ INITIALIZATION ~~~ ###
def _download_nltk_data():
    """
    Ensures all necessary NLTK models are available locally.
    We explicitly list all dependencies for the NER pipeline.
    """
    required_packages = [
        "punkt",  # Tokenizer
        "averaged_perceptron_tagger",  # POS Tagger (generic)
        "averaged_perceptron_tagger_eng",  # POS Tagger (English specific)
        "maxent_ne_chunker",  # NE Chunker Base
        "maxent_ne_chunker_tab",  # NE Chunker Tabular Model
        "words",  # Word list for valid English words
        "treebank",  # Training data reference
    ]

    for package in required_packages:
        nltk.download(package, quiet=True, raise_on_error=False)


_download_nltk_data()


### ~~~ FUNCTIONS ~~~ ###
def extract_entities_nltk(
    text: str, valid_airports: FrozenSet[str], valid_aircraft: FrozenSet[str]
) -> List[Entity]:
    """
    Extracts entities using NLTK for Named Entity Recognition (NER),
    then validates candidates against the reference data. This is done
    for both Airports and Aircraft, follwing these steps:
        1. Tokenization: Split text into tokens/words.
        2. POS Tagging: Assign Part-of-Speech tags to each token.
        3. NER Chunking: Group tokens into Named Entities.
        4. Extraction & Validation: Extract relevant entities and validate.
    Args:
        text: Raw user text.
        valid_airports: Reference set for validation.
        valid_aircraft: Reference set for validation.

    Returns:
        List[Entity]: List of NLTK-discovered validated entities.
    """
    entities = []

    ### 1. Tokenize: Split text into words/tokens ###
    tokens = word_tokenize(text)

    ### 2. POS Tag: Assign Part-of-Speech tags (Noun, Verb, etc.) ###
    tagged = pos_tag(tokens)

    ### 3. Chunk: Group tokens into Named Entities (Person, Organization, GPE) ###
    chunked = ne_chunk(tagged)

    ### 4. Extract & Validate Entities ###
    for i in chunked:
        if isinstance(i, nltk.tree.Tree):
            """
            NLTK represents Named Entities as subtrees.
            Each subtree has a label (entity type) and leaves (the actual words).
            we can extract both for further processing.
            """
            ## This is a Named Entity (e.g., (GPE New/NNP York/NNP)) ##
            entity_type = i.label()
            entity_text = " ".join([token for token, tag in i.leaves()])

            ## Logic: Airport Extraction ##
            if entity_type in ("GPE", "ORGANIZATION"):
                """
                GPE (Geo-Political Entity) and ORGANIZATION are common NER labels
                that might contain airport codes or names. We apply heuristics to filter
                potential airport codes from these entities.
                """
                # Heuristic: If it looks like a 3-letter code and is in our DB, take it. #
                if (
                    re.fullmatch(r"[A-Z]{3}", entity_text.upper())
                    and entity_text.upper() in valid_airports
                ):
                    entities.append(
                        Entity(entity_type="AIRPORT", value=entity_text.upper())
                    )
        else:
            """
            NLTK represents regular tokens as tuples (word, tag).
            We can scan these tokens for aircraft models.
            """
            ## This is a regular tuple (word, tag) ##
            token, _ = i

            ## Logic: Aircraft Extraction ##
            for model in valid_aircraft:
                """
                Aircraft models are irregular and don't have a consistent NER label.
                We perform substring matching against our reference set, as a fallback.
                """
                # Check if the known model name is contained within the token #
                if model.upper() in token.upper():
                    # Avoid duplicates if already added
                    new_ent = Entity(entity_type="AIRCRAFT", value=model)
                    if new_ent not in entities:
                        entities.append(new_ent)

    return entities


def process_query_nltk(raw_query: str, ref_data: ReferenceData) -> ProcessedQuery:
    """
    Main pipeline function for the NLTK/Probabilistic Approach.
    Args:
        raw_query: The user's original input string.
        ref_data: The immutable ground truth data.

    Returns:
        ProcessedQuery: The structured representation.
    """
    ### 1. Run NLTK Extraction ###
    nltk_entities = extract_entities_nltk(
        raw_query, ref_data.airports, ref_data.aircraft_models
    )

    ### 2. Run Metric Extraction (Keyword based, NLTK not needed for this simple task) ###
    metric_entities = extract_metrics(raw_query)

    ### 3. Merge and Deduplicate ###
    entities = []
    entities.extend(nltk_entities)
    entities.extend(metric_entities)

    ### 3.5. Deduplication Logic ###
    unique_entities = []
    seen: Set[Tuple[str, str]] = set()
    for ent in entities:
        key = (ent.entity_type, ent.value)
        if key not in seen:
            unique_entities.append(ent)
            seen.add(key)

    # 4. Determine Intent
    intent = determine_intent(raw_query, unique_entities)

    return ProcessedQuery(
        original_text=raw_query, intent=intent, entities=unique_entities
    )

### ~~~ GLOBAL IMPORTS ~~~ ###

### ~~~ LOCAL IMPORTS ~~~ ###
from src.components.ingestion.processor_regex import process_query_regex
from src.components.ingestion.processor_nltk import process_query_nltk
from src.components.ingestion.processor import load_reference_data, ReferenceData


def run_comparison():
    """
    Executes the comparison test case.
    It injects 'JFK' into the reference data temporarily to demonstrate extraction capabilities
    even when the underlying CSV might be missing that specific airport.
    """
    # 1. Load Ground Truth
    csv_path = "dbs/Airline_surveys_sample.csv"
    original_ref_data = load_reference_data(csv_path)

    # 2. Prepare Test Data
    # The user specifically requested a test involving "JFK".
    # Since "JFK" might not be in the sample CSV, we manually inject it
    # into the reference set for a fair test of the *extraction logic*.
    modified_airports = set(original_ref_data.airports)
    modified_airports.add("JFK")

    modified_ref_data = ReferenceData(
        airports=frozenset(modified_airports),
        aircraft_models=original_ref_data.aircraft_models,
    )

    test_query = "Show me flights from JFK to LAX on a B777-200 and their delay stats"

    print(f'--- Running Comparison for Query: "{test_query}" ---\n')

    # 3. Execute Regex Approach
    print("= Regex Approach =")
    regex_result = process_query_regex(test_query, modified_ref_data)
    print(f"Original Text: {regex_result.original_text}")
    print(f"Intent:        {regex_result.intent}")
    print(f"Entities:      {regex_result.entities}")
    print("-" * 30)

    # 4. Execute NLTK Approach
    print("= NLTK Approach =")
    nltk_result = process_query_nltk(test_query, modified_ref_data)
    print(f"Original Text: {nltk_result.original_text}")
    print(f"Intent:        {nltk_result.intent}")
    print(f"Entities:      {nltk_result.entities}")
    print("-" * 30)

    # 5. Analyze Differences
    regex_airports = [
        e.value for e in regex_result.entities if e.entity_type == "AIRPORT"
    ]
    nltk_airports = [
        e.value for e in nltk_result.entities if e.entity_type == "AIRPORT"
    ]

    print("\n--- Summary ---")
    print(f"Regex Airports Found: {regex_airports}")
    print(f"NLTK Airports Found:  {nltk_airports}")

    # Heuristic check for the specific 'JFK' case
    has_jfk_regex = "JFK" in regex_airports
    has_jfk_nltk = "JFK" in nltk_airports

    if has_jfk_nltk and not has_jfk_regex:
        print("\nResult: NLTK successfully outperformed Regex on 'JFK'.")
    elif has_jfk_nltk and has_jfk_regex:
        print(
            "\nResult: Both approaches found 'JFK' (likely due to reference injection)."
        )
    elif not has_jfk_nltk:
        print("\nResult: NLTK failed to find 'JFK' (investigate NLTK chunking).")


if __name__ == "__main__":
    run_comparison()

### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import List, Callable, TypeAlias
from dataclasses import dataclass

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import (
    ProcessedQuery,
    IntentType,
    Entity,
    EntityType,
    ContextChunk,
    pretty_print,
)
from src.retrieval.graph_retrieval import query_graph_cypher


### ~~~ CUSTOM TYPES ~~~ ###
ValidatorFunc: TypeAlias = Callable[[List[ContextChunk]], bool]


@dataclass
class GraphTestCase:
    description: str
    query: ProcessedQuery
    validator: ValidatorFunc


### ~~~ STATE DEFINITIONS ~~~ ###
def has_results(results: List[ContextChunk]) -> bool:
    return len(results) > 0


def is_empty(results: List[ContextChunk]) -> bool:
    return len(results) == 0


TEST_CASES: List[GraphTestCase] = [
    # --- 1. FLIGHT_SEARCH ---
    GraphTestCase(
        description="Flight Search: All Flights",
        query=ProcessedQuery(
            original_text="Show me all flights",
            intent=IntentType.FLIGHT_SEARCH,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Flight Search: From LAX",
        query=ProcessedQuery(
            original_text="Show flights departing from LAX",
            intent=IntentType.FLIGHT_SEARCH,
            entities=[Entity(EntityType.AIRPORT, "LAX")],
        ),
        validator=has_results,
    ),
    # --- 2. DELAY_ANALYSIS ---
    GraphTestCase(
        description="Delay Analysis: General",
        query=ProcessedQuery(
            original_text="Which flights have the most delays?",
            intent=IntentType.DELAY_ANALYSIS,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Delay Analysis: Specific Flight (757)",
        query=ProcessedQuery(
            original_text="How delayed is flight 757?",
            intent=IntentType.DELAY_ANALYSIS,
            entities=[Entity(EntityType.FLIGHT_NUMBER, "757")],
        ),
        validator=has_results,
    ),
    # --- 3. SATISFACTION_METRICS ---
    GraphTestCase(
        description="Satisfaction Metrics: General",
        query=ProcessedQuery(
            original_text="What is the passenger satisfaction score?",
            intent=IntentType.SATISFACTION_METRICS,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Satisfaction Metrics: By Generation (Boomer)",
        query=ProcessedQuery(
            original_text="Are Boomers satisfied?",
            intent=IntentType.SATISFACTION_METRICS,
            entities=[Entity(EntityType.GENERATION, "Boomer")],
        ),
        # Assuming "Baby Boomer" exists, if not it might fail or return empty.
        # Using has_results tentatively, expecting some data.
        validator=has_results,
    ),
    # --- 4. ROUTE_STATS ---
    GraphTestCase(
        description="Route Stats: Busiest Routes",
        query=ProcessedQuery(
            original_text="What are the busiest routes?",
            intent=IntentType.ROUTE_STATS,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Route Stats: Route Popularity",
        query=ProcessedQuery(
            original_text="Show me route popularity",
            intent=IntentType.ROUTE_STATS,
            entities=[],
        ),
        validator=has_results,
    ),
    # --- 5. LOYALTY_ANALYSIS ---
    GraphTestCase(
        description="Loyalty Analysis: Compare Levels",
        query=ProcessedQuery(
            original_text="How do loyalty members compare in miles flown?",
            intent=IntentType.LOYALTY_ANALYSIS,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Loyalty Analysis: Specific Level (premier platinum)",
        query=ProcessedQuery(
            original_text="Stats for premier platinum members",
            intent=IntentType.LOYALTY_ANALYSIS,
            entities=[Entity(EntityType.LOYALTY_LEVEL, "premier platinum")],
        ),
        validator=has_results,
    ),
    # --- 6. DEMOGRAPHIC_INSIGHTS ---
    GraphTestCase(
        description="Demographic Insights: General",
        query=ProcessedQuery(
            original_text="Show demographic breakdown",
            intent=IntentType.DEMOGRAPHIC_INSIGHTS,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Demographic Insights: Gen X",
        query=ProcessedQuery(
            original_text="Insights for Gen X",
            intent=IntentType.DEMOGRAPHIC_INSIGHTS,
            entities=[Entity(EntityType.GENERATION, "Gen X")],
        ),
        validator=has_results,
    ),
    # --- 7. FEEDBACK_VOLUME ---
    GraphTestCase(
        description="Feedback Volume: Most Feedback",
        query=ProcessedQuery(
            original_text="Which flights have the most feedback?",
            intent=IntentType.FEEDBACK_VOLUME,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Feedback Volume: Specific Flight (42)",
        query=ProcessedQuery(
            original_text="Feedback count for flight 42",
            intent=IntentType.FEEDBACK_VOLUME,
            entities=[Entity(EntityType.FLIGHT_NUMBER, "42")],
        ),
        validator=has_results,
    ),
    # --- 8. FLEET_PERFORMANCE ---
    GraphTestCase(
        description="Fleet Performance: All Types",
        query=ProcessedQuery(
            original_text="Which aircraft types have the best performance?",
            intent=IntentType.FLEET_PERFORMANCE,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Fleet Performance: Specific Aircraft (B737-800)",
        query=ProcessedQuery(
            original_text="Performance of B737-800",
            intent=IntentType.FLEET_PERFORMANCE,
            entities=[Entity(EntityType.AIRCRAFT, "B737-800")],
        ),
        validator=has_results,
    ),
    # --- 9. CABIN_CLASS_STATS ---
    GraphTestCase(
        description="Cabin Class Stats: Compare Classes",
        query=ProcessedQuery(
            original_text="Compare cabin classes",
            intent=IntentType.CABIN_CLASS_STATS,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Cabin Class Stats: Economy",
        query=ProcessedQuery(
            original_text="Stats for Economy Class",
            intent=IntentType.CABIN_CLASS_STATS,
            entities=[Entity(EntityType.CABIN_CLASS, "Economy")],
        ),
        validator=has_results,
    ),
    # --- 10. CONNECTION_STATS ---
    GraphTestCase(
        description="Connection Stats: Direct vs Connecting",
        query=ProcessedQuery(
            original_text="Compare direct vs connecting flights",
            intent=IntentType.CONNECTION_STATS,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Connection Stats: Layover Analysis",
        query=ProcessedQuery(
            original_text="Analyze layover impact",
            intent=IntentType.CONNECTION_STATS,
            entities=[],
        ),
        validator=has_results,
    ),
    # --- 11. AIRPORT_STATS ---
    GraphTestCase(
        description="Airport Stats: General",
        query=ProcessedQuery(
            original_text="Show airport statistics",
            intent=IntentType.AIRPORT_STATS,
            entities=[],
        ),
        validator=has_results,
    ),
    GraphTestCase(
        description="Airport Stats: Specific Airport (LAX)",
        query=ProcessedQuery(
            original_text="Stats for LAX airport",
            intent=IntentType.AIRPORT_STATS,
            entities=[Entity(EntityType.AIRPORT, "LAX")],
        ),
        validator=has_results,
    ),
    # --- 12. UNKNOWN ---
    GraphTestCase(
        description="Unknown Intent: CEO",
        query=ProcessedQuery(
            original_text="Who is the CEO?",
            intent=IntentType.UNKNOWN,
            entities=[],
        ),
        validator=is_empty,
    ),
    GraphTestCase(
        description="Unknown Intent: Weather",
        query=ProcessedQuery(
            original_text="What is the weather like?",
            intent=IntentType.UNKNOWN,
            entities=[],
        ),
        validator=is_empty,
    ),
]

MAX_CHUNKS_TO_DISPLAY: int = 3
MAX_TEXT_LENGTH_FOR_DISPLAY: int = 100


### ~~~ FUNCTION DEFINITIONS ~~~ ###
def main() -> int:
    """"""
    error_count = 0
    print(f"Running {len(TEST_CASES)} graph retrieval tests...\n")

    for test_case in TEST_CASES:
        print(f"--- Testing: {test_case.description} ---")
        try:
            results = query_graph_cypher(test_case.query)

            if not test_case.validator(results):
                print(f"Test failed for query: {test_case.query.original_text}")
                print(
                    f"Expected: Validator '{test_case.validator.__name__}' to return True"
                )
                print(f"Got: {len(results)} results")
                if results:
                    print("First result sample:")
                    pretty_print(results[0])
                print("-" * 40)
                error_count += 1
            else:
                print(f"Success. Found {len(results)} chunks.")
                if results:
                    print(f"--- Displaying first {MAX_CHUNKS_TO_DISPLAY} chunks ---")
                    for i, chunk in enumerate(results[:MAX_CHUNKS_TO_DISPLAY]):
                        truncated_text = (
                            chunk.text[:MAX_TEXT_LENGTH_FOR_DISPLAY] + "..."
                            if len(chunk.text) > MAX_TEXT_LENGTH_FOR_DISPLAY
                            else chunk.text
                        )
                        print(f"  Chunk {i + 1}:")
                        print(f"    id: {chunk.id}")
                        print(f'    text: "{truncated_text}"')
                        print(f"    source: {chunk.source}")
                    print("-" * 40)
        except Exception as e:
            print(f"Test failed with exception: {e}")
            error_count += 1
        print("")

    if error_count == 0:
        print("All tests passed!")
    else:
        print(f"{error_count / len(TEST_CASES) * 100:.1f}% tests failed.")
    return 0


### ~~~ SCRIPT EXECUTION ~~~ ###
if __name__ == "__main__":
    main()

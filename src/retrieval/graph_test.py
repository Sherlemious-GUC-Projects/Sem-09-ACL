from src.utils.types import ProcessedQuery, IntentType
from graph_retrieval import query_graph_cypher
import pprint


def test_flight_search_all():
    print("\n--- TEST: Flight Search (All Flights) ---")
    query = ProcessedQuery(
        original_text="Show me all flights",
        intent=IntentType.FLIGHT_SEARCH,
        entities=[],
    )
    results = query_graph_cypher(query)
    print(f"Found {len(results)} flights")
    for chunk in results[:3]:  # Show first 3 only
        pprint.pprint(chunk)


def test_route_stats():
    print("\n--- TEST: Route Stats (Top Routes) ---")
    query = ProcessedQuery(
        original_text="What are the busiest routes?",
        intent=IntentType.ROUTE_STATS,
        entities=[],
    )
    results = query_graph_cypher(query)
    print(f"Found {len(results)} routes")
    for chunk in results:
        pprint.pprint(chunk)


def test_delay_analysis():
    print("\n--- TEST: Delay Analysis ---")
    query = ProcessedQuery(
        original_text="Which flights have the most delays?",
        intent=IntentType.DELAY_ANALYSIS,
        entities=[],
    )
    results = query_graph_cypher(query)
    print(f"Found {len(results)} flight delay records")
    for chunk in results[:3]:
        pprint.pprint(chunk)


def test_loyalty_analysis():
    print("\n--- TEST: Loyalty Analysis ---")
    query = ProcessedQuery(
        original_text="How do loyalty members compare in miles flown?",
        intent=IntentType.LOYALTY_ANALYSIS,
        entities=[],
    )
    results = query_graph_cypher(query)
    print(f"Found {len(results)} loyalty levels")
    for chunk in results:
        pprint.pprint(chunk)


def test_fleet_performance():
    print("\n--- TEST: Fleet Performance ---")
    query = ProcessedQuery(
        original_text="Which aircraft types have the best performance?",
        intent=IntentType.FLEET_PERFORMANCE,
        entities=[],
    )
    results = query_graph_cypher(query)
    print(f"Found {len(results)} fleet types")
    for chunk in results:
        pprint.pprint(chunk)


def test_connection_stats():
    print("\n--- TEST: Connection Stats ---")
    query = ProcessedQuery(
        original_text="Compare direct vs connecting flights",
        intent=IntentType.CONNECTION_STATS,
        entities=[],
    )
    results = query_graph_cypher(query)
    print(f"Found {len(results)} connection types")
    for chunk in results:
        pprint.pprint(chunk)


def test_unknown_intent():
    print("\n--- TEST: Unknown Intent ---")
    query = ProcessedQuery(
        original_text="Who is the CEO?", intent=IntentType.UNKNOWN, entities=[]
    )
    results = query_graph_cypher(query)
    print(f"Results: {results} (Expected: [])")


if __name__ == "__main__":
    test_flight_search_all()
    test_route_stats()
    test_delay_analysis()
    test_loyalty_analysis()
    test_fleet_performance()
    test_connection_stats()
    test_unknown_intent()

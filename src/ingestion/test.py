### ~~~ GLOBAL IMPORTS ~~~ ###
# None

### ~~~ LOCAL IMPORTS ~~~ ###
from utils.types import IntentType, Entity, ProcessedQuery, equals, pretty_print
from ingestion.main import main as ingestion_main


### ~~~ STATE DEFINITIONS ~~~ ###
TEST_QUERIES: list[ProcessedQuery] = [
    ProcessedQuery(
        original_text=(
            "I need options for getting to London next Tuesday "
            "that land before noon at LHR."
        ),
        intent=IntentType.FLIGHT_SEARCH,
        entities=[
            Entity("DATE", "next Tuesday"),
            Entity("AIRPORT", "LHR"),
        ],
    ),
    ProcessedQuery(
        original_text=(
            "Is the punctuality of the morning service to Chicago "
            "getting worse in the winter on flights into ORD?"
        ),
        intent=IntentType.DELAY_ANALYSIS,
        entities=[
            Entity("AIRPORT", "ORD"),
            Entity("DATE", "winter"),
        ],
    ),
    ProcessedQuery(
        original_text=(
            "Are the seats in the new A321neo configuration "
            "actually comfortable according to recent surveys?"
        ),
        intent=IntentType.SATISFACTION_METRICS,
        entities=[
            Entity("AIRCRAFT", "A321neo"),
        ],
    ),
    ProcessedQuery(
        original_text=(
            "How does the A380's maintenance downtime compare "
            "to the 787 Dreamliner over the past year?"
        ),
        intent=IntentType.FLEET_PERFORMANCE,
        entities=[
            Entity("AIRCRAFT", "A380"),
            Entity("AIRCRAFT", "787 Dreamliner"),
            Entity("DATE", "past year"),
        ],
    ),
    ProcessedQuery(
        original_text="Can you book a hotel for me near the arrival gate at CDG?",
        intent=IntentType.UNKNOWN,
        entities=[
            Entity("AIRPORT", "CDG"),
        ],
    ),
]


def main() -> int:
    """"""
    error_count = 0
    for test_query in TEST_QUERIES:
        result = ingestion_main(test_query.original_text)
        if not equals(result, test_query):
            print("Test failed for query:", test_query.original_text)
            print("Expected:")
            pretty_print(test_query)
            print("Got:")
            pretty_print(result)
            print("-" * 40)
            error_count += 1

    if error_count == 0:
        print("All tests passed!")
    else:
        print(f"{error_count / len(TEST_QUERIES) * 100}% tests failed.")
    return 0


if __name__ == "__main__":
    main()
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/

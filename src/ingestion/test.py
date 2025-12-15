### ~~~ GLOBAL IMPORTS ~~~ ###
# None

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import IntentType, Entity, ProcessedQuery


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
    return 0


if __name__ == "__main__":
    main()
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/

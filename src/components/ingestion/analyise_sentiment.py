### ~~~ GLOBAL IMPORTS ~~~ ###
from transformers import pipeline

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import IntentType

### ~~~ STATE DEFINITIONS ~~~ ###
CANDIDATE_LABELS = [
    "search for flights and itineraries",
    "flight delays and punctuality",
    "passenger satisfaction and service ratings",
    "flight routes and city pairs",
    "frequent flyer programs and loyalty",
    "passenger demographics and age groups",
    "volume of customer feedback and complaints",
    "aircraft models and fleet performance",
    "cabin classes and seat categories",
    "connecting flights and layovers",
    "specific airport performance",
    "other general request",
]

LABEL_TO_INTENT = {
    "search for flights and itineraries": IntentType.FLIGHT_SEARCH,
    "flight delays and punctuality": IntentType.DELAY_ANALYSIS,
    "passenger satisfaction and service ratings": IntentType.SATISFACTION_METRICS,
    "flight routes and city pairs": IntentType.ROUTE_STATS,
    "frequent flyer programs and loyalty": IntentType.LOYALTY_ANALYSIS,
    "passenger demographics and age groups": IntentType.DEMOGRAPHIC_INSIGHTS,
    "volume of customer feedback and complaints": IntentType.FEEDBACK_VOLUME,
    "aircraft models and fleet performance": IntentType.FLEET_PERFORMANCE,
    "cabin classes and seat categories": IntentType.CABIN_CLASS_STATS,
    "connecting flights and layovers": IntentType.CONNECTION_STATS,
    "specific airport performance": IntentType.AIRPORT_STATS,
    "other general request": IntentType.UNKNOWN,
}


CONFIDENCE_THRESHOLD: float = 0.3

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def analyze_intent(raw_query: str) -> IntentType:
    """"""
    ### run the model ###
    classification = classifier(
        raw_query,
        candidate_labels=CANDIDATE_LABELS,
    )

    ### unpack results ###
    top_label: str = classification["labels"][0]  # type: ignore[reportIndexIssue]
    score: float = classification["scores"][0]  # type: ignore[reportIndexIssue]

    ### set a confidence threshold ###
    if score < CONFIDENCE_THRESHOLD:
        return IntentType.UNKNOWN

    return LABEL_TO_INTENT[top_label]


def main() -> int:
    """"""
    test_queries_and_intents = [
        (
            "I need options for getting to London next Tuesday that land before noon.",
            IntentType.FLIGHT_SEARCH,
        ),
        (
            "Is the punctuality of the morning service to Chicago getting worse in the winter?",
            IntentType.DELAY_ANALYSIS,
        ),
        (
            "Are the seats in the new configuration actually comfortable according to the recent surveys?",
            IntentType.SATISFACTION_METRICS,
        ),
        (
            "Which city pair has seen the highest passenger volume increase this quarter?",
            IntentType.ROUTE_STATS,
        ),
        (
            "Do frequent flyers actually rate the lounge access higher than casual travelers?",
            IntentType.LOYALTY_ANALYSIS,
        ),
        (
            "Are Millennial travelers prioritizing Wi-Fi access more than Boomers?",
            IntentType.DEMOGRAPHIC_INSIGHTS,
        ),
        (
            "Have we seen a spike in complaints regarding the new check-in kiosk?",
            IntentType.FEEDBACK_VOLUME,
        ),
        (
            "How does the A380's maintenance downtime compare to the Dreamliner?",
            IntentType.FLEET_PERFORMANCE,
        ),
        (
            "Is the price premium for the front of the plane justified by better service scores?",
            IntentType.CABIN_CLASS_STATS,
        ),
        (
            "What's the probability of lost luggage on itineraries with tight layovers vs non-stop?",
            IntentType.CONNECTION_STATS,
        ),
        (
            "Is the congestion at LAX affecting departure times more than weather?",
            IntentType.AIRPORT_STATS,
        ),
        ("Can you book a hotel for me near the arrival gate?", IntentType.UNKNOWN),
    ]
    count = 0
    for query, expected_intent in test_queries_and_intents:
        detected_intent = analyze_intent(query)
        if detected_intent != expected_intent:
            print(
                f"Test failed for query: '{query}'. Expected: {expected_intent}, Detected: {detected_intent}"
            )
            count += 1
    if count == 0:
        print("All tests passed!")
    else:
        print(
            f"{count} tests failed. thats {count / len(test_queries_and_intents) * 100}% failure rate"
        )

    return 0


if __name__ == "__main__":
    main()

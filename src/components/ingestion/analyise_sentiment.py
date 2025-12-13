### ~~~ GLOBAL IMPORTS ~~~ ###
from transformers import pipeline

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import IntentType

### ~~~ STATE DEFINITIONS ~~~ ###
CANDIDATE_LABELS = [
    "finding and booking specific flights",
    "analysis of delay causes and punctuality trends",
    "general passenger satisfaction scores and survey ratings",
    "route-specific passenger volume and statistics",
    "analysis of frequent flyer segments vs casual travelers",
    "analysis of passenger demographics and generations",
    "volume of customer complaints and feedback",
    "performance metrics of specific aircraft models",
    "comparison of value and service in premium vs economy cabin classes",
    "statistics on connecting flights vs direct flights",
    "operational status and congestion at specific airports",
    "request for hotel booking or non-flight services",
]

LABEL_TO_INTENT = {
    "finding and booking specific flights": IntentType.FLIGHT_SEARCH,
    "analysis of delay causes and punctuality trends": IntentType.DELAY_ANALYSIS,
    "general passenger satisfaction scores and survey ratings": IntentType.SATISFACTION_METRICS,
    "route-specific passenger volume and statistics": IntentType.ROUTE_STATS,
    "analysis of frequent flyer segments vs casual travelers": IntentType.LOYALTY_ANALYSIS,
    "analysis of passenger demographics and generations": IntentType.DEMOGRAPHIC_INSIGHTS,
    "volume of customer complaints and feedback": IntentType.FEEDBACK_VOLUME,
    "performance metrics of specific aircraft models": IntentType.FLEET_PERFORMANCE,
    "comparison of value and service in premium vs economy cabin classes": IntentType.CABIN_CLASS_STATS,
    "statistics on connecting flights vs direct flights": IntentType.CONNECTION_STATS,
    "operational status and congestion at specific airports": IntentType.AIRPORT_STATS,
    "request for hotel booking or non-flight services": IntentType.UNKNOWN,
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

### ~~~ GLOBAL IMPORTS ~~~ ###
from transformers import pipeline

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import IntentType
from src.utils.constant import CONFIDENCE_THRESHOLD, CANDIDATE_LABELS, LABEL_TO_INTENT

### ~~~ STATE DEFINITIONS ~~~ ###
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

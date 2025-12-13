from src.utils.types import IntentType

test_queries_and_intents = [
    # FLIGHT_SEARCH (Current: "finding and booking specific flights")
    (
        "I need options for getting to London next Tuesday that land before noon.",
        IntentType.FLIGHT_SEARCH,
    ),
    (
        "Look up availability for a one-way to Tokyo on the 25th.",
        IntentType.FLIGHT_SEARCH,
    ),
    (
        "Can I get a seat on the redeye to New York next Friday?",
        IntentType.FLIGHT_SEARCH,
    ),
    # DELAY_ANALYSIS (Current: "analysis of delay causes and punctuality trends")
    (
        "Is the punctuality of the morning service to Chicago getting worse in the winter?",
        IntentType.DELAY_ANALYSIS,
    ),
    (
        "Why is flight AA123 constantly arriving behind schedule this month?",
        IntentType.DELAY_ANALYSIS,
    ),
    (
        "Identify the primary reasons for the late departures from Heathrow.",
        IntentType.DELAY_ANALYSIS,
    ),
    # SATISFACTION_METRICS (Current: "general passenger satisfaction scores and survey ratings")
    (
        "Are the seats in the new configuration actually comfortable according to the recent surveys?",
        IntentType.SATISFACTION_METRICS,
    ),
    (
        "How are passengers rating the new in-flight meal options?",
        IntentType.SATISFACTION_METRICS,
    ),
    (
        "What is the average Net Promoter Score for the long-haul fleet?",
        IntentType.SATISFACTION_METRICS,
    ),
    # ROUTE_STATS (Current: "route-specific passenger volume and statistics")
    (
        "Which city pair has seen the highest passenger volume increase this quarter?",
        IntentType.ROUTE_STATS,
    ),
    (
        "Show me the load factor trends for the Dubai to London corridor.",
        IntentType.ROUTE_STATS,
    ),
    (
        "Identify the most profitable routes operated by the airline last year.",
        IntentType.ROUTE_STATS,
    ),
    # LOYALTY_ANALYSIS (Current: "analysis of frequent flyer segments vs casual travelers")
    (
        "Do frequent flyers actually rate the lounge access higher than casual travelers?",
        IntentType.LOYALTY_ANALYSIS,
    ),
    (
        "Are Gold tier members flying more frequently than Silver tier members?",
        IntentType.LOYALTY_ANALYSIS,
    ),
    (
        "What is the retention rate for our top-tier loyalty program members?",
        IntentType.LOYALTY_ANALYSIS,
    ),
    # DEMOGRAPHIC_INSIGHTS (Current: "analysis of passenger demographics and generations")
    (
        "Are Millennial travelers prioritizing Wi-Fi access more than Boomers?",
        IntentType.DEMOGRAPHIC_INSIGHTS,
    ),
    (
        "Do families with children prefer evening or morning departures?",
        IntentType.DEMOGRAPHIC_INSIGHTS,
    ),
    (
        "What is the breakdown of business travelers versus leisure travelers by age group?",
        IntentType.DEMOGRAPHIC_INSIGHTS,
    ),
    # FEEDBACK_VOLUME (Current: "volume of customer complaints and feedback")
    (
        "Have we seen a spike in complaints regarding the new check-in kiosk?",
        IntentType.FEEDBACK_VOLUME,
    ),
    (
        "Count the number of negative reviews mentioning 'rude staff' in the last week.",
        IntentType.FEEDBACK_VOLUME,
    ),
    (
        "Is there an increase in positive feedback after the recent policy change?",
        IntentType.FEEDBACK_VOLUME,
    ),
    # FLEET_PERFORMANCE (Current: "performance metrics of specific aircraft models")
    (
        "How does the A380's maintenance downtime compare to the Dreamliner?",
        IntentType.FLEET_PERFORMANCE,
    ),
    (
        "Which aircraft type has the highest fuel efficiency per passenger mile?",
        IntentType.FLEET_PERFORMANCE,
    ),
    (
        "Compare the technical reliability of the Boeing 737 MAX versus the Airbus A320neo.",
        IntentType.FLEET_PERFORMANCE,
    ),
    # CABIN_CLASS_STATS (Current: "comparison of value and service in premium vs economy cabin classes")
    (
        "Is the price premium for the front of the plane justified by better service scores?",
        IntentType.CABIN_CLASS_STATS,
    ),
    (
        "Are business class passengers experiencing fewer delays than economy passengers?",
        IntentType.CABIN_CLASS_STATS,
    ),
    (
        "What is the difference in meal satisfaction between First Class and Economy?",
        IntentType.CABIN_CLASS_STATS,
    ),
    # CONNECTION_STATS (Current: "statistics on connecting flights vs direct flights")
    (
        "What's the probability of lost luggage on itineraries with tight layovers vs non-stop?",
        IntentType.CONNECTION_STATS,
    ),
    (
        "Do direct flights have a significantly higher on-time performance than connecting ones?",
        IntentType.CONNECTION_STATS,
    ),
    (
        "Analyze the missed connection rate for passengers transiting through Frankfurt.",
        IntentType.CONNECTION_STATS,
    ),
    # AIRPORT_STATS (Current: "operational status and congestion at specific airports")
    (
        "Is the congestion at LAX affecting departure times more than weather?",
        IntentType.AIRPORT_STATS,
    ),
    (
        "How does the baggage handling speed at JFK compare to Newark?",
        IntentType.AIRPORT_STATS,
    ),
    (
        "Report on the average taxi-out time for flights leaving O'Hare.",
        IntentType.AIRPORT_STATS,
    ),
    # UNKNOWN (Current: "request for hotel booking or non-flight services")
    ("Can you book a hotel for me near the arrival gate?", IntentType.UNKNOWN),
    ("Where is the best place to get coffee near the office?", IntentType.UNKNOWN),
    ("Translate 'Hello' into Spanish for me.", IntentType.UNKNOWN),
]

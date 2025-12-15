from .types import IntentType

### ~~~ INGESTION ~~~ ###
CSV_PATH = "dbs/Airline_surveys_sample.csv"
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

### ~~~ KG ~~~ ###
CONFIG_PATH = "./config.txt"

### ~~~ IR ~~~ ###
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "airline_reviews"
USE_OLLAMA = True

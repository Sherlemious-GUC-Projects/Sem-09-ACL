from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any

class IntentType(str, Enum):
    FLIGHT_SEARCH = "FLIGHT_SEARCH"
    DELAY_ANALYSIS = "DELAY_ANALYSIS"
    SATISFACTION_METRICS = "SATISFACTION"
    ROUTE_STATS = "ROUTE_STATS"
    LOYALTY_ANALYSIS = "LOYALTY_ANALYSIS"
    DEMOGRAPHIC_INSIGHTS = "DEMOGRAPHIC"
    FEEDBACK_VOLUME = "FEEDBACK_VOLUME"
    FLEET_PERFORMANCE = "FLEET_PERFORMANCE"
    CABIN_CLASS_STATS = "CABIN_CLASS_STATS"
    CONNECTION_STATS = "CONNECTION_STATS"
    AIRPORT_STATS = "AIRPORT_STATS"
    UNKNOWN = "UNKNOWN"

@dataclass
class Entity:
    entity_type: str  # "AIRPORT", "FLIGHT_NUM", "DATE", "AIRCRAFT", "METRIC"
    value: str        # "ORD", "BA123", "2023-01-01", "737 Max", "Food"

@dataclass
class ProcessedQuery:
    original_text: str
    intent: IntentType
    entities: List[Entity]

class RetrievalSource(str, Enum):
    CYPHER = "CYPHER"
    VECTOR = "VECTOR"

@dataclass
class ContextChunk:
    id: str                 # Unique ID of the node/record (e.g., "Journey_12345")
    text: str               # Human-readable info (e.g., "Journey 123 had 45min delay, Food Score: 2")
    score: float            # Relevance score (1.0 for Cypher exact match, 0.0-1.0 for Vector)
    source: RetrievalSource # Where did this come from?
    metadata: Dict[str, Any] # Raw props for UI visualization


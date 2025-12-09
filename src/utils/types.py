from dataclasses import dataclass
from enum import Enum
from typing import List, Dict


class IntentType(str, Enum):
    FLIGHT_SEARCH = "FLIGHT_SEARCH"
    DELAY_ANALYSIS = "DELAY_ANALYSIS"
    SATISFACTION_METRICS = "SATISFACTION"
    ROUTE_STATS = "ROUTE_STATS"
    UNKNOWN = "UNKNOWN"


@dataclass
class Entity:
    entity_type: str  # "AIRPORT", "FLIGHT_NUM", "DATE", "AIRCRAFT", "METRIC"
    value: str


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
    id: str
    text: str
    score: float
    source: RetrievalSource
    metadata: Dict

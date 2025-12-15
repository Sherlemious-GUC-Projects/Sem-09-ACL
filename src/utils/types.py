from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

##########################
### ~~~ MAIN TYPES ~~~ ###
##########################


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
    entity_type: str  # "AIRPORT", "FLIGHT_NUM", "DATE", "AIRCRAFT"
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


############################
### ~~~ UILITY TYPES ~~~ ###
############################


@dataclass
class Config:
    uri: str
    username: str
    password: str


#################################
### ~~~ UTILITY FUNCTIONS ~~~ ###
#################################


def pretty_print(object: Entity | ProcessedQuery | ContextChunk) -> None:
    """"""
    attrs = vars(object)
    print(f"{object.__class__.__name__}:")
    for key, value in attrs.items():
        print(f"  {key}: {value}")
    print()


def equals(
    obj1: Entity | ProcessedQuery | ContextChunk,
    obj2: Entity | ProcessedQuery | ContextChunk,
) -> bool:
    """"""
    if obj1.__class__ != obj2.__class__:
        return False
    return vars(obj1) == vars(obj2)

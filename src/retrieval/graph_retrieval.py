### ~~~ GLOBAL IMPORTS ~~~ ###
from neo4j import GraphDatabase, Driver
from typing import List


### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import (
    IntentType,
    ProcessedQuery,
    ContextChunk,
    RetrievalSource,
    EntityType,
)
from .util import load_config, Config

# ~~~ CYPHER TEMPLATES ~~~
TEMPLATE_MAP = {
    IntentType.FLIGHT_SEARCH: """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(o:Airport)
        MATCH (f)-[:ARRIVES_AT]->(d:Airport)
        WHERE (o.station_code = $origin OR $origin IS NULL)
          AND (d.station_code = $destination OR $destination IS NULL)
          AND (toString(f.flight_number) = $flight_number OR $flight_number IS NULL)
          AND (f.fleet_type_description = $aircraft OR $aircraft IS NULL)
        RETURN 
            "Flight " + f.flight_number + " from " + o.station_code + " to " + d.station_code + 
            " on " + f.fleet_type_description AS text,
            f.flight_number AS id,
            {flight: f.flight_number, origin: o.station_code, dest: d.station_code, fleet: f.fleet_type_description} AS metadata
        LIMIT 10
    """,
    IntentType.DELAY_ANALYSIS: """
        MATCH (j:Journey)-[:ON]->(f:Flight)
        WHERE (toString(f.flight_number) = $flight_number OR $flight_number IS NULL)
        WITH f, avg(j.arrival_delay_minutes) as avg_delay, count(j) as count
        ORDER BY avg_delay DESC
        RETURN
            "Flight " + f.flight_number + " has an average delay of " + toString(round(avg_delay, 1)) + 
            " minutes based on " + toString(count) + " journeys." AS text,
            f.flight_number + "_delay" AS id,
            {flight: f.flight_number, avg_delay: avg_delay, journey_count: count} AS metadata
        LIMIT 5
    """,
    IntentType.SATISFACTION_METRICS: """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WITH p, avg(j.food_satisfaction_score) as avg_food_score
        WHERE (p.generation = $generation OR $generation IS NULL)
        RETURN 
            "Passengers of generation " + p.generation + " have an average food satisfaction score of " + 
            toString(round(avg_food_score, 1)) AS text,
            p.generation + "_satisfaction" AS id,
            {generation: p.generation, avg_food: avg_food_score} AS metadata
        ORDER BY avg_food_score DESC
        LIMIT 5
    """,
    IntentType.ROUTE_STATS: """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(o:Airport)
        MATCH (f)-[:ARRIVES_AT]->(d:Airport)
        WITH o.station_code + "-" + d.station_code as route, count(f) as flight_count
        RETURN 
            "Route " + route + " has " + toString(flight_count) + " flights." AS text,
            route + "_stats" AS id,
            {route: route, flights: flight_count} AS metadata
        ORDER BY flight_count DESC
        LIMIT 5
    """,
    IntentType.LOYALTY_ANALYSIS: """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE (p.loyalty_program_level = $loyalty_level OR $loyalty_level IS NULL)
        WITH p.loyalty_program_level AS loyalty_level, 
             avg(j.actual_flown_miles) AS avg_miles,
             count(DISTINCT p) AS passenger_count
        RETURN 
            "Loyalty level " + loyalty_level + " passengers fly an average of " + 
            toString(round(avg_miles, 1)) + " miles (" + toString(passenger_count) + " passengers)." AS text,
            loyalty_level + "_loyalty" AS id,
            {loyalty_level: loyalty_level, avg_miles: avg_miles, passengers: passenger_count} AS metadata
        ORDER BY avg_miles DESC
        LIMIT 5
    """,
    IntentType.DEMOGRAPHIC_INSIGHTS: """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE (p.generation = $generation OR $generation IS NULL)
        WITH p.generation AS generation,
             count(j) AS journey_count,
             avg(j.food_satisfaction_score) AS avg_food,
             avg(j.arrival_delay_minutes) AS avg_delay
        RETURN 
            "Generation " + generation + ": " + toString(journey_count) + " journeys, " +
            "avg food score " + toString(round(avg_food, 1)) + ", avg delay " + 
            toString(round(avg_delay, 1)) + " min." AS text,
            generation + "_demographic" AS id,
            {generation: generation, journeys: journey_count, avg_food: avg_food, avg_delay: avg_delay} AS metadata
        ORDER BY journey_count DESC
        LIMIT 5
    """,
    IntentType.FEEDBACK_VOLUME: """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
        WHERE (toString(f.flight_number) = $flight_number OR $flight_number IS NULL)
        WITH f.flight_number AS flight_id, count(j) AS feedback_count
        RETURN 
            "Flight " + toString(flight_id) + " has " + toString(feedback_count) + " feedback entries." AS text,
            toString(flight_id) + "_feedback" AS id,
            {flight: flight_id, feedback_count: feedback_count} AS metadata
        ORDER BY feedback_count DESC
        LIMIT 10
    """,
    IntentType.FLEET_PERFORMANCE: """
        MATCH (j:Journey)-[:ON]->(f:Flight)
        WHERE (f.fleet_type_description = $aircraft OR $aircraft IS NULL)
        WITH f.fleet_type_description AS fleet_type,
             avg(j.arrival_delay_minutes) AS avg_delay,
             avg(j.food_satisfaction_score) AS avg_food,
             count(j) AS journey_count
        RETURN 
            "Fleet " + fleet_type + ": avg delay " + toString(round(avg_delay, 1)) + " min, " +
            "avg food score " + toString(round(avg_food, 1)) + " (" + toString(journey_count) + " journeys)." AS text,
            fleet_type + "_fleet" AS id,
            {fleet: fleet_type, avg_delay: avg_delay, avg_food: avg_food, journeys: journey_count} AS metadata
        ORDER BY journey_count DESC
        LIMIT 5
    """,
    IntentType.CABIN_CLASS_STATS: """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE (j.passenger_class = $cabin_class OR $cabin_class IS NULL)
        WITH j.passenger_class AS cabin_class,
             count(j) AS journey_count,
             avg(j.food_satisfaction_score) AS avg_food,
             avg(j.arrival_delay_minutes) AS avg_delay
        RETURN 
            "Cabin class " + cabin_class + ": " + toString(journey_count) + " journeys, " +
            "avg food score " + toString(round(avg_food, 1)) + ", avg delay " + 
            toString(round(avg_delay, 1)) + " min." AS text,
            cabin_class + "_cabin" AS id,
            {cabin_class: cabin_class, journeys: journey_count, avg_food: avg_food, avg_delay: avg_delay} AS metadata
        ORDER BY journey_count DESC
        LIMIT 5
    """,
    IntentType.CONNECTION_STATS: """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WITH CASE WHEN j.number_of_legs > 1 THEN "Multi-leg" ELSE "Direct" END AS connection_type,
             count(j) AS journey_count,
             avg(j.arrival_delay_minutes) AS avg_delay,
             avg(j.food_satisfaction_score) AS avg_food
        RETURN 
            connection_type + " journeys: " + toString(journey_count) + " total, " +
            "avg delay " + toString(round(avg_delay, 1)) + " min, " +
            "avg food score " + toString(round(avg_food, 1)) + "." AS text,
            connection_type + "_connection" AS id,
            {type: connection_type, journeys: journey_count, avg_delay: avg_delay, avg_food: avg_food} AS metadata
        ORDER BY journey_count DESC
    """,
    IntentType.AIRPORT_STATS: """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(a:Airport)
        MATCH (j:Journey)-[:ON]->(f)
        WHERE (a.station_code = $origin OR $origin IS NULL)
        WITH a.station_code AS airport,
             count(DISTINCT f) AS flight_count,
             avg(j.arrival_delay_minutes) AS avg_delay
        RETURN 
            "Airport " + airport + ": " + toString(flight_count) + " departing flights, " +
            "avg delay " + toString(round(avg_delay, 1)) + " min." AS text,
            airport + "_airport" AS id,
            {airport: airport, flights: flight_count, avg_delay: avg_delay} AS metadata
        ORDER BY flight_count DESC
        LIMIT 5
    """,
    # Fallback for unknown intents
    IntentType.UNKNOWN: "",
}


def get_driver() -> Driver:
    """
    Creates a Neo4j driver using the shared config.
    Args:
        None
    Returns:
        Driver: Neo4j driver instance.
    """
    config: Config = load_config()
    return GraphDatabase.driver(config.uri, auth=(config.username, config.password))


def query_graph_cypher(processed_input: ProcessedQuery) -> List[ContextChunk]:
    """
    Executes a Cypher query based on the user's intent and entities.
    Args:
        processed_input (ProcessedQuery): The processed user query with intent and entities.
    Returns:
        List[ContextChunk]: Retrieved context chunks from the graph database.
    """
    if (
        processed_input.intent not in TEMPLATE_MAP
        or not TEMPLATE_MAP[processed_input.intent]
    ):
        return []

    # 1. Extract parameters from entities
    params: dict[str, str | None] = {
        "origin": None,
        "destination": None,
        "flight_number": None,
        "aircraft": None,
        "generation": None,
        "metric": None,
        "loyalty_level": None,
        "cabin_class": None,
    }

    for entity in processed_input.entities:
        if entity.entity_type == EntityType.AIRPORT:
            # Simple heuristic: first airport is origin, second is dest (refine later if needed)
            if not params["origin"]:
                params["origin"] = entity.value
            else:
                params["destination"] = entity.value
        elif entity.entity_type == EntityType.FLIGHT_NUMBER:
            params["flight_number"] = entity.value
        elif entity.entity_type == EntityType.AIRCRAFT:
            params["aircraft"] = entity.value
        elif entity.entity_type == EntityType.GENERATION:
            params["generation"] = entity.value
        elif entity.entity_type == EntityType.LOYALTY_LEVEL:
            params["loyalty_level"] = entity.value
        elif entity.entity_type == EntityType.CABIN_CLASS:
            params["cabin_class"] = entity.value

    # 2. Get the Cypher template
    query = TEMPLATE_MAP[processed_input.intent]

    results = []
    driver = get_driver()

    try:
        with driver.session() as session:
            result = session.run(query, params)

            for record in result:
                chunk = ContextChunk(
                    id=str(record["id"]),
                    text=record["text"],
                    score=1.0,
                    source=RetrievalSource.CYPHER,
                    metadata=record["metadata"],
                )
                results.append(chunk)
    except Exception as e:
        print(f"Error executing Cypher query: {e}")
    finally:
        driver.close()

    return results

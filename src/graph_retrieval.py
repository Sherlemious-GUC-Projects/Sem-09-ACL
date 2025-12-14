from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from models.models import IntentType, ProcessedQuery, ContextChunk, RetrievalSource, Entity
from util import load_config, Config

# ~~~ CYPHER TEMPLATES ~~~
TEMPLATE_MAP = {
    IntentType.FLIGHT_SEARCH: """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(o:Airport)
        MATCH (f)-[:ARRIVES_AT]->(d:Airport)
        WHERE (o.station_code = $origin OR $origin IS NULL)
          AND (d.station_code = $destination OR $destination IS NULL)
          AND (f.flight_number = $flight_number OR $flight_number IS NULL)
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
        WHERE (f.flight_number = $flight_number OR $flight_number IS NULL)
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

    # Fallback for unknown intents
    IntentType.UNKNOWN: ""
}

def get_driver() -> Driver:
    """
    Creates a Neo4j driver using the shared config.
    """
    config: Config = load_config()
    return GraphDatabase.driver(config.uri, auth=(config.username, config.password))

def query_graph_cypher(processed_input: ProcessedQuery) -> List[ContextChunk]:
    """
    Executes a Cypher query based on the user's intent and entities.
    """
    if processed_input.intent not in TEMPLATE_MAP or not TEMPLATE_MAP[processed_input.intent]:
        return []

    # 1. Extract parameters from entities
    params = {
        "origin": None,
        "destination": None,
        "flight_number": None,
        "aircraft": None,
        "generation": None,
        "metric": None
    }
    
    for entity in processed_input.entities:
        if entity.entity_type == "AIRPORT":
            # Simple heuristic: first airport is origin, second is dest (refine later if needed)
            if not params["origin"]:
                params["origin"] = entity.value
            else:
                params["destination"] = entity.value
        elif entity.entity_type == "FLIGHT_NUM":
            params["flight_number"] = entity.value
        elif entity.entity_type == "AIRCRAFT":
            params["aircraft"] = entity.value
        elif entity.entity_type == "GENERATION":
            params["generation"] = entity.value

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
                    metadata=record["metadata"]
                )
                results.append(chunk)
    except Exception as e:
        print(f"Error executing Cypher query: {e}")
    finally:
        driver.close()
        
    return results


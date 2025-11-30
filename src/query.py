### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import List, Dict, Callable


def query_1(session) -> List[Dict]:
    """
    Identify the 5 airport-to-airport routes with the highest number of flights.
    Returns:
        List[Dict]: Each dict contains origin, destination, and flight_count.
    """
    result = session.run(
        """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(o:Airport)
        MATCH (f)-[:ARRIVES_AT]->(d:Airport)
        WITH o.station_code AS origin,
             d.station_code AS destination,
             count(f) AS flight_count
        RETURN origin, destination, flight_count
        ORDER BY flight_count DESC
        LIMIT 5
        """
    )

    rows: List[Dict] = [
        {
            "origin": record["origin"],
            "destination": record["destination"],
            "flight_count": record["flight_count"],
        }
        for record in result
    ]

    return rows


def query_2(session) -> List[Dict]:
    """
    Identify the top 10 Flights with the most passenger feedback.
    Returns:
        List[Dict]: Each dict contains flight_id and feedback_count.
    """
    result = session.run(
        """
        MATCH (:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
        WITH f.flight_number AS flight_id,
             count(j) AS feedback_count
        RETURN flight_id, feedback_count
        ORDER BY feedback_count DESC
        LIMIT 10
        """
    )

    rows: List[Dict] = [
        {
            "flight_id": record["flight_id"],
            "feedback_count": record["feedback_count"],
        }
        for record in result
    ]

    return rows


def query_3(session) -> List[Dict]:
    """
    Calculate average food satisfaction for multi-leg journeys grouped by generation.
    Returns:
        List[Dict]: Each dict contains generation, multi_leg_count, avg_score.
    """
    result = session.run(
        """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE j.number_of_legs > 1
        WITH p.generation AS generation, j
        RETURN generation,
               count(j) AS multi_leg_count,
               avg(j.food_satisfaction_score) AS avg_score
        ORDER BY multi_leg_count DESC
        """
    )

    rows: List[Dict] = [
        {
            "generation": record["generation"],
            "multi_leg_count": record["multi_leg_count"],
            "avg_score": record["avg_score"],
        }
        for record in result
    ]

    return rows


def query_4(session) -> List[Dict]:
    """
    Calculate average arrival delay per flight and return the 10 with the shortest delays.
    Returns:
        List[Dict]: Each dict contains flight_id and avg_arrival_delay.
    """
    result = session.run(
        """
        MATCH (j:Journey)-[:ON]->(f:Flight)
        WITH f.flight_number AS flight_id,
             avg(toFloat(j.arrival_delay_minutes)) AS avg_arrival_delay
        RETURN flight_id, avg_arrival_delay
        ORDER BY avg_arrival_delay ASC
        LIMIT 10
        """
    )

    rows: List[Dict] = [
        {
            "flight_id": record["flight_id"],
            "avg_arrival_delay": record["avg_arrival_delay"],
        }
        for record in result
    ]

    return rows


def query_5(session) -> List[Dict]:
    """
    Calculate the average flown miles for passengers grouped by loyalty program level.
    Returns:
        List[Dict]: Each dict contains loyalty_level and avg_actual_flown_miles.
    """
    result = session.run(
        """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WITH p.loyalty_program_level AS loyalty_level,
             avg(j.actual_flown_miles) AS avg_actual_flown_miles
        RETURN loyalty_level, avg_actual_flown_miles
        ORDER BY avg_actual_flown_miles DESC
        """
    )

    rows: List[Dict] = [
        {
            "loyalty_level": record["loyalty_level"],
            "avg_actual_flown_miles": record["avg_actual_flown_miles"],
        }
        for record in result
    ]

    return rows


query_mapper: Dict[int, Callable] = {
    1: query_1,
    2: query_2,
    3: query_3,
    4: query_4,
    5: query_5,
}

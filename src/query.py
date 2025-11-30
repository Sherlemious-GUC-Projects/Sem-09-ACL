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


def query_6(session) -> List[Dict]:
    """
    Count passengers whose overall satisfaction score exceeds 3.
    Returns:
        List[Dict]: Single dict with key 'satisfied_count'.
    """
    result = session.run(
        """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WITH p,
             j.food_satisfaction_score AS food,
             j.arrival_delay_minutes AS arrival_delay_minutes,
             j.number_of_legs AS number_of_legs,
             j.actual_flown_miles AS actual_flown_miles
        WITH p,
             food,
             CASE
               WHEN round(abs(arrival_delay_minutes) / 20.0, 1) > 5 THEN 0.0
               WHEN round(abs(arrival_delay_minutes) / 20.0, 1) < 0 THEN 5.0
               ELSE round(5 - round(abs(arrival_delay_minutes) / 20.0, 1), 1)
             END AS delay_score,
             CASE
               WHEN round(number_of_legs * 1.5, 1) > 5 THEN 0.0
               WHEN round(number_of_legs * 1.5, 1) < 0 THEN 5.0
               ELSE round(5 - round(number_of_legs * 1.5, 1), 1)
             END AS legs_score,
             CASE
               WHEN round(actual_flown_miles / 3000.0, 1) > 5 THEN 0.0
               WHEN round(actual_flown_miles / 3000.0, 1) < 0 THEN 5.0
               ELSE round(5 - round(actual_flown_miles / 3000.0, 1), 1)
             END AS miles_score
        WITH p,
             food,
             delay_score,
             legs_score,
             miles_score,
             round(
               (0.5 * food) +
               (0.35 * delay_score) +
               (0.1 * legs_score) +
               (0.05 * miles_score),
               1
             ) AS overall_satisfaction_score
        WHERE overall_satisfaction_score > 3
        RETURN count(DISTINCT p) AS satisfied_count
        """
    )

    rows: List[Dict] = [
        {
            "satisfied_count": record["satisfied_count"],
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
    6: query_6,
}

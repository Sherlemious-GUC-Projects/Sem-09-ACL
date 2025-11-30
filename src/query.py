### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import List, Dict


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

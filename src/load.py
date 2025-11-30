### ~~~ GLOBAL IMPORTS ~~~ ###
from typing import Dict, List

### ~~~ LOCAL IMPORTS ~~~ ###
from util import chunk_rows


def load_passengers(session, rows: List[Dict], chunk_size: int = 1_000) -> None:
    """
    Load Passenger nodes.
    """
    for chunk in chunk_rows(rows, chunk_size=chunk_size):
        session.execute_write(
            lambda tx, c=chunk: tx.run(
                """
                UNWIND $rows AS row
                MERGE (p:Passenger {record_locator: row.record_locator})
                  SET p.loyalty_program_level = row.loyalty_program_level,
                      p.generation = row.generation
                """,
                rows=c,
            )
        )


def load_journeys(session, rows: List[Dict], chunk_size: int = 1_000) -> None:
    """
    Load Journey nodes.
    """
    for chunk in chunk_rows(rows, chunk_size=chunk_size):
        session.execute_write(
            lambda tx, c=chunk: tx.run(
                """
                UNWIND $rows AS row
                MERGE (j:Journey {feedback_ID: row.feedback_ID})
                  SET j.food_satisfaction_score = row.food_satisfaction_score,
                      j.arrival_delay_minutes = row.arrival_delay_minutes,
                      j.actual_flown_miles = row.actual_flown_miles,
                      j.number_of_legs = row.number_of_legs,
                      j.passenger_class = row.passenger_class
                """,
                rows=c,
            )
        )


def load_flights(session, rows: List[Dict], chunk_size: int = 1_000) -> None:
    """
    Load Flight nodes.
    """
    for chunk in chunk_rows(rows, chunk_size=chunk_size):
        session.execute_write(
            lambda tx, c=chunk: tx.run(
                """
                UNWIND $rows AS row
                MERGE (f:Flight {flight_number: row.flight_number,
                                 fleet_type_description: row.fleet_type_description})
                """,
                rows=c,
            )
        )


def load_airports(session, rows: List[Dict], chunk_size: int = 1_000) -> None:
    """
    Load Airport nodes.
    """
    for chunk in chunk_rows(rows, chunk_size=chunk_size):
        session.execute_write(
            lambda tx, c=chunk: tx.run(
                """
                UNWIND $rows AS row
                MERGE (a:Airport {station_code: row.station_code})
                """,
                rows=c,
            )
        )


def load_passenger_journey_rels(
    session, rows: List[Dict], chunk_size: int = 1_000
) -> None:
    """
    Load Passenger-TOOK->Journey relationships.
    """
    for chunk in chunk_rows(rows, chunk_size=chunk_size):
        session.execute_write(
            lambda tx, c=chunk: tx.run(
                """
                UNWIND $rows AS row
                MERGE (p:Passenger {record_locator: row.record_locator})
                MERGE (j:Journey {feedback_ID: row.feedback_ID})
                MERGE (p)-[:TOOK]->(j)
                """,
                rows=c,
            )
        )


def load_journey_flight_rels(
    session, rows: List[Dict], chunk_size: int = 1_000
) -> None:
    """
    Load Journey-ON->Flight relationships.
    """
    for chunk in chunk_rows(rows, chunk_size=chunk_size):
        session.execute_write(
            lambda tx, c=chunk: tx.run(
                """
                UNWIND $rows AS row
                MERGE (j:Journey {feedback_ID: row.feedback_ID})
                MERGE (f:Flight {flight_number: row.flight_number,
                                 fleet_type_description: row.fleet_type_description})
                MERGE (j)-[:ON]->(f)
                """,
                rows=c,
            )
        )


def load_flight_airport_rels(
    session, rows: List[Dict], chunk_size: int = 1_000
) -> None:
    """
    Load Flight to Airport relationships.
    """
    for chunk in chunk_rows(rows, chunk_size=chunk_size):
        session.execute_write(
            lambda tx, c=chunk: tx.run(
                """
                UNWIND $rows AS row
                MERGE (f:Flight {flight_number: row.flight_number,
                                 fleet_type_description: row.fleet_type_description})
                MERGE (o:Airport {station_code: row.origin_station_code})
                MERGE (d:Airport {station_code: row.destination_station_code})
                MERGE (f)-[:DEPARTS_FROM]->(o)
                MERGE (f)-[:ARRIVES_AT]->(d)
                """,
                rows=c,
            )
        )

### ~~~ GLOBALS IMPORTS ~~~ ###
from neo4j import GraphDatabase

### ~~~ LOCAL IMPORTS ~~~ ###
from util import (
    Config,
    load_config,
    load_data,
)
from construct import (
    construct_passanger_data,
    construct_journey_data,
    construct_flight_data,
    construct_airport_data,
    construct_passenger_journey_rels,
    construct_journey_flight_rels,
    construct_flight_airport_rels,
)
from load import (
    load_passengers,
    load_journeys,
    load_flights,
    load_airports,
    load_passenger_journey_rels,
    load_journey_flight_rels,
    load_flight_airport_rels,
)


def apply_constraints(session) -> None:
    """
    Ensure uniqueness constraints exist for all node identifiers.
    Args:
        session: An open Neo4j session.
    Returns:
        None
    """
    constraint_statements = [
        """
        CREATE CONSTRAINT passenger_record_locator IF NOT EXISTS
        FOR (p:Passenger) REQUIRE p.record_locator IS UNIQUE
        """,
        """
        CREATE CONSTRAINT journey_feedback_id IF NOT EXISTS
        FOR (j:Journey) REQUIRE j.feedback_ID IS UNIQUE
        """,
        """
        CREATE CONSTRAINT airport_station_code IF NOT EXISTS
        FOR (a:Airport) REQUIRE a.station_code IS UNIQUE
        """,
        """
        CREATE CONSTRAINT flight_number_fleet IF NOT EXISTS
        FOR (f:Flight) REQUIRE (f.flight_number, f.fleet_type_description) IS UNIQUE
        """,
    ]

    for statement in constraint_statements:
        session.execute_write(lambda tx: tx.run(statement))


def delete_non_schema_nodes(session) -> None:
    """
    Delete any nodes that do not belong to the defined schema labels.
    Args:
        session: An open Neo4j session.
    Returns:
        None
    """
    session.execute_write(
        lambda tx: tx.run(
            """
            MATCH (n)
            WHERE NONE(lbl IN labels(n) WHERE lbl IN ['Passenger','Journey','Flight','Airport'])
            DETACH DELETE n
            """
        )
    )


def create_driver(config: Config):
    """
    Create a Neo4j driver using provided config.
    Args:
        config (Config): Configuration for connecting to Neo4j.
    Returns:
        neo4j.Driver: A Neo4j driver instance.
    """
    driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))
    return driver


def loader() -> int:
    """
    Entry point for creating the Knowledge Graph in Neo4j.
    Returns:
        int: Status code.
    """
    ### load config ###
    config = load_config()

    ### load data ###
    df = load_data()

    ### build derived datasets (kept for potential inspection/debugging) ###
    passenger_data = construct_passanger_data(df)
    journey_data = construct_journey_data(df)
    flight_data = construct_flight_data(df)
    airport_data = construct_airport_data(df)
    passenger_journey_rels = construct_passenger_journey_rels(df)
    journey_flight_rels = construct_journey_flight_rels(df)
    flight_airport_rels = construct_flight_airport_rels(df)

    ### create driver and load data ###
    driver = create_driver(config)
    try:
        with driver.session() as session:
            apply_constraints(session)
            delete_non_schema_nodes(session)
            load_passengers(session, passenger_data.to_dict(orient="records"))
            load_journeys(session, journey_data.to_dict(orient="records"))
            load_flights(session, flight_data.to_dict(orient="records"))
            load_airports(session, airport_data.to_dict(orient="records"))
            load_passenger_journey_rels(
                session, passenger_journey_rels.to_dict(orient="records")
            )
            load_journey_flight_rels(
                session, journey_flight_rels.to_dict(orient="records")
            )
            load_flight_airport_rels(
                session, flight_airport_rels.to_dict(orient="records")
            )
    finally:
        driver.close()

    return 0


def main() -> int:
    """
    Main function to execute the loader.
    Returns:
        int: Status code.
    """
    ### load data ###
    loader()

    return 0


if __name__ == "__main__":
    exit(main())

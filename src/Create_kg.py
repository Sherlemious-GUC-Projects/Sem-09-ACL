### ~~~ GLOBALS IMPORTS ~~~ ###
from typing import Dict, Iterable, List, Sequence
from dataclasses import dataclass
from neo4j import GraphDatabase
import pandas as pd
import json as js
import os


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass
class Config:
    uri: str
    username: str
    password: str


#################################
### ~~~ UTILITY FUNCTIONS ~~~ ###
#################################


def load_config(path: str = "./config.txt") -> Config:
    """
    Load configuration from a file, returning a Config instance.
    Args:
        path (str): Path to the configuration file.
    Returns:
        Config: An instance of the Config dataclass with loaded
                configuration values.
    Throws:
        KeyError: If any required keys are missing in the config file.
    """
    ### read config file ###
    with open(path, "r") as file:
        lines = file.readlines()

    ### parse config file ###
    config_dict = {}
    for line in lines:
        key, value = line.strip().split("=")
        config_dict[key] = value

    ### insure all keys are present ###
    required_keys = {"URI", "USERNAME", "PASSWORD"}
    if not required_keys.issubset(config_dict.keys()):
        missing_keys = required_keys - config_dict.keys()
        raise KeyError(f"Missing keys in config file: {missing_keys}")

    ### create Config instance ###
    config: Config = Config(
        uri=config_dict["URI"],
        username=config_dict["USERNAME"],
        password=config_dict["PASSWORD"],
    )

    return config


def load_data(path: str = "./dbs/Airline_surveys_sample.csv") -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    Args:
        path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    Throws:
        FileNotFoundError: If the specified file does not exist.
    """
    ### insure file exists ###
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at path: {path}")

    ### load data ###
    df: pd.DataFrame = pd.read_csv(path)

    return df


def chunk_rows(rows: Sequence[Dict], chunk_size: int = 1_000) -> Iterable[List[Dict]]:
    """
    Yield successive chunks from a list of dictionaries.
    Args:
        rows (Sequence[Dict]): All rows to be chunked.
        chunk_size (int): Maximum number of rows per chunk.
    Returns:
        Iterable[List[Dict]]: Generator yielding list chunks.
    """
    for start in range(0, len(rows), chunk_size):
        end = start + chunk_size
        yield list(rows[start:end])


#####################################
### ~~~ CONSTRUCTOR FUNCTIONS ~~~ ###
#####################################


def construct_passanger_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct passenger data from the given DataFrame.
    1. extracts their record_locator
    2. extracts their loyalty_program_level
    3. and finally adds the generation
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing record_locator,
                      loyalty_program_level, and generation.
    Throws:
        None
    """
    ### get all unique passengers ###
    unique_passengers: pd.DataFrame = (
        df[["record_locator", "loyalty_program_level", "generation"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return unique_passengers


def construct_journey_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct journey data from the given DataFrame.
    1. for each journey, extracts relevant journey attributes
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing feedback_ID,
                      food_satisfaction_score, arrival_delay_minutes,
                      actual_flown_miles, number_of_legs, passenger_class.
    Throws:
        None
    """
    ### get all unique journeys ###
    unique_journeys: pd.DataFrame = df[
        [
            "feedback_ID",
            "food_satisfaction_score",
            "arrival_delay_minutes",
            "actual_flown_miles",
            "number_of_legs",
            "passenger_class",
        ]
    ].copy()

    ### remove duplicates ###
    unique_journeys = unique_journeys.drop_duplicates().reset_index(drop=True)

    return unique_journeys


def construct_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct flight data from the given DataFrame.
    1. for each flight, create a unique identifier
    2. extracts relevant flight attributes
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing flight_number,
                      fleet_type_description.
    Throws:
        None
    """
    ### get all unique flights ###
    unique_flights: pd.DataFrame = df[
        [
            "flight_number",
            "fleet_type_description",
            "origin_station_code",
            "destination_station_code",
        ]
    ].copy()

    ### remove duplicates ###
    unique_flights = unique_flights.drop_duplicates().reset_index(drop=True)

    return unique_flights


def construct_airport_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct airport data from the given DataFrame.
    1. for each airport, extracts relevant airport attributes
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing station_code,
                      airport_name, city, state, country.
    Throws:
        None
    """
    ### get all unique airports ###
    origin_airports: pd.DataFrame = df[["origin_station_code"]].copy()
    destination_airports: pd.DataFrame = df[["destination_station_code"]].copy()

    unique_airports: pd.DataFrame = pd.concat(
        [
            origin_airports.rename(columns={"origin_station_code": "station_code"}),
            destination_airports.rename(
                columns={"destination_station_code": "station_code"}
            ),
        ]
    ).drop_duplicates()

    unique_airports = unique_airports.reset_index(drop=True)

    return unique_airports


def construct_passenger_journey_rels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct passenger-to-journey relationship data.
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing record_locator and feedback_ID.
    """
    rels: pd.DataFrame = (
        df[["record_locator", "feedback_ID"]].drop_duplicates().reset_index(drop=True)
    )
    return rels


def construct_journey_flight_rels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct journey-to-flight relationship data.
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing feedback_ID, flight_number,
                      and fleet_type_description.
    """
    rels: pd.DataFrame = (
        df[["feedback_ID", "flight_number", "fleet_type_description"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return rels


def construct_flight_airport_rels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct flight-to-airport relationship data.
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing flight_number, fleet_type_description,
                      origin_station_code, destination_station_code.
    """
    rels: pd.DataFrame = (
        df[
            [
                "flight_number",
                "fleet_type_description",
                "origin_station_code",
                "destination_station_code",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return rels


################################
### ~~~ LOADER FUNCTIONS ~~~ ###
################################


def load_query_answer(id: int) -> List[Dict]:
    """
    Load the expected answer for a given query from a JSON file.
    Args:
        id (int): The query ID.
    Returns:
        List[Dict]: The expected answer as a list of dictionaries.
    Throws:
        FileNotFoundError: If the expected answer file does not exist.
    """
    path = f"./dbs/query_{id}.json"

    ### insure file exists ###
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected answer file not found at path: {path}")

    ### load expected answer ###
    with open(path, "r", encoding="utf-8-sig") as file:
        answer: List[Dict] = js.load(file)

    return answer


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


##############################
### ~~~ MAIN FUNCTIONS ~~~ ###
##############################


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
    loader()

    return 0


if __name__ == "__main__":
    exit(main())

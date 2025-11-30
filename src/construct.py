### ~~~ GLOBALS IMPORTS ~~~ ###
import pandas as pd


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

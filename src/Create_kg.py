### ~~~ GLOBAL IMPORTS ~~~ ###
from dataclasses import dataclass
import pandas as pd
import uuid
import os


### ~~~ LOCAL IMPORTS ~~~ ###
# None


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass
class Config:
    uri: str
    username: str
    password: str


### ~~~ STATE MANAGEMENT ~~~ ###
# None


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


def construct_passanger_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct passenger data from the given DataFrame.
    1. for each passenger, create a unique identifier
    2. extracts as well their loyalty_program_level
    3. and finally adds the generation
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing passenger_id,
                      loyalty_program_level, and generation.
    Throws:
        None
    Note:
        An important assumption here is that each row in the dataframe
        corresponds to a unique passenger.
    """
    ### get all unique passengers ###
    unique_passengers: pd.DataFrame = df[["loyalty_program_level", "generation"]].copy()

    ### add unique identifier ###
    unique_passengers["passenger_id"] = [
        str(uuid.uuid4()) for _ in range(unique_passengers.shape[0])
    ]

    return unique_passengers


def construct_journey_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct journey data from the given DataFrame.
    1. for each journey, create a unique identifier
    2. extracts relevant journey attributes
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

    ### add unique identifier ###

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
        ["flight_number", "fleet_type_description"]
    ].copy()

    ### add unique identifier ###

    return unique_flights


def construct_airport_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct airport data from the given DataFrame.
    1. for each airport, create a unique identifier
    2. extracts relevant airport attributes
    Args:
        df (pd.DataFrame): DataFrame containing the original data.
    Returns:
        pd.DataFrame: DataFrame containing station_code,
                      airport_name, city, state, country.
    Throws:
        None
    """
    ### get all unique airports ###
    origin_airports: pd.DataFrame = df[
        [
            "origin_station_code",
        ]
    ].copy()
    destination_airports: pd.DataFrame = df[
        [
            "destination_station_code",
        ]
    ].copy()

    unique_airports: pd.DataFrame = (
        pd.concat([origin_airports, destination_airports])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return unique_airports


def main() -> int:
    """ """
    ### load config ###
    # config = load_config()

    ### load data ###
    df = load_data()

    ### construct passenger data ###
    passenger_data = construct_passanger_data(df)

    ### construct journey data ###
    journey_data = construct_journey_data(df)

    ### construct flight data ###
    flight_data = construct_flight_data(df)

    ### construct airport data ###
    airport_data = construct_airport_data(df)

    return 0


if __name__ == "__main__":
    exit(main())

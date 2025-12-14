### ~~~ GLOBALS IMPORTS ~~~ ###
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import pandas as pd
import json as js
import os


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass
class Config:
    uri: str
    username: str
    password: str


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


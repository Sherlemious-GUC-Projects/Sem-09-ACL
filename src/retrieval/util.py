### ~~~ GLOBALS IMPORTS ~~~ ###
from typing import Iterable, Sequence
import pandas as pd
import json as js
import os


### ~~~ GLOBALS IMPORTS ~~~ ###
from utils.types import Config
from utils.constant import CSV_PATH, CONFIG_PATH


def load_config(path: str = CONFIG_PATH) -> Config:
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
        if line == "" or line.startswith("#"):
            continue
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


def load_data(path: str = CSV_PATH) -> pd.DataFrame:
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


def chunk_rows(rows: Sequence[dict], chunk_size: int = 1_000) -> Iterable[list[dict]]:
    """
    Yield successive chunks from a list of dictionaries.
    Args:
        rows (Sequence[dict]): All rows to be chunked.
        chunk_size (int): Maximum number of rows per chunk.
    Returns:
        Iterable[list[dict]]: Generator yielding list chunks.
    """
    for start in range(0, len(rows), chunk_size):
        end = start + chunk_size
        yield list(rows[start:end])


def load_query_answer(id: int) -> list[dict]:
    """
    Load the expected answer for a given query from a JSON file.
    Args:
        id (int): The query ID.
    Returns:
        list[dict]: The expected answer as a list of dictionaries.
    Throws:
        FileNotFoundError: If the expected answer file does not exist.
    """
    path = f"./dbs/query_{id}.json"

    ### insure file exists ###
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected answer file not found at path: {path}")

    ### load expected answer ###
    with open(path, "r", encoding="utf-8-sig") as file:
        answer: list[dict] = js.load(file)

    return answer

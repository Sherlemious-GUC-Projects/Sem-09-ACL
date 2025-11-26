### ~~~ GLOBAL IMPORTS ~~~ ###
from dataclasses import dataclass
import pandas as pd
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


def main() -> int:
    """ """
    ### load config ###
    config = load_config()

    ### load data ###
    df = load_data()

    print(df.head())
    return 0


if __name__ == "__main__":
    exit(main())

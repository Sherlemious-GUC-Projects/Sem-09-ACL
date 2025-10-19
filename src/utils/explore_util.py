### ~~~ GLOBAL IMPORTS ~~~ ###
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
import pandas as pd
import pathlib
import json

### ~~~ LOCAL IMPORTS ~~~ ###
# None

### ~~~ STATE DEFINITION ~~~ ###
COLS = [
    # passenger data
    "Passanger_Name",
    "Rating",
    "Verified",
    "Review_title",
    "Review_content",
    "Traveller_Type",
    "Class",
    # spatial data
    "Flying_Date",
    "Layover_Route",
    "Route",
    "Start_Location",
    "End_Location",
    "Start_Latitude",
    "Start_Longitude",
    "Start_Address",
    "End_Latitude",
    "End_Longitude",
    "End_Address",
]
COLS_PASSENGER = COLS[0:7]
COLS_SPATIAL = COLS[7:]


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    Args:
        path, str: The file path to the CSV file.
    Returns:
        df, pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    ### insure the path exists ###
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"The file at path {path} does not exist.")

    df: pd.DataFrame = pd.read_csv(path)

    return df


def get_sentiment_scores(sentence: str) -> tuple[float, int]:
    """
    Compute sentiment scores for a given sentence using VADER.
    Args:
        sentence, str: The input sentence to analyze.
    Returns:
        A tuple of the following:
            - compound sentiment score (float)
            - interpretation (int): 1 for positive, -1 for negative, 0 for neutral
    """
    ### init the analyzer ###
    analyzer: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
    score: dict = analyzer.polarity_scores(sentence)
    interpretation: int = (
        1 if score["compound"] > 0.05 else (-1 if score["compound"] < -0.05 else 0)
    )
    return score["compound"], interpretation


def main() -> int:
    """"""
    ### init some stuff ###
    input_path: str = "./dbs/raw/db.csv"

    ### load the data ###
    df: pd.DataFrame = load_data(input_path)

    return 0


if __name__ == "__main__":
    exit(main())

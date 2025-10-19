### ~~~ GLOBAL IMPORTS ~~~ ###
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

### ~~~ LOCAL IMPORTS ~~~ ###
from explore_util import (
    load_data,
    get_sentiment_scores,
    COLS,
    COLS_PASSENGER,
    COLS_SPATIAL,
)


def process_passenger_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process passenger-related data. It does the following:
        - Dense encoding of the 'Class' column. (possibly apply an exponential growth encoding later)
        - Sparse encoding of the 'Traveller_Type' column.
        - Sparse encoding of the 'Verified' column.
        - Frequency encoding of the 'Passanger_Name' column.
        - Sentiment analysis on the 'Review_content' column.
        - Drops the 'Review_title' column.
        - Drops any remaining object-type columns. (which should not be ANY)
    Args:
        df, pd.DataFrame: The input DataFrame containing passenger data.
    Returns:
        df, pd.DataFrame: The processed DataFrame.
    """
    ### copy the df to avoid modifying the original ###
    df = df.copy()

    ### do a dense enocding of the class column ###
    mapper: dict[str, int] = {
        "Economy Class": 0,
        "Business Class": 1,
        "Premium Economy": 2,
        "First Class": 3,
        "Unknown": 4,
    }
    df.loc["Class_Encoded"] = df["Class"].map(mapper)
    df.drop(columns=["Class"], inplace=True)

    ### do a sparese encoding of the traveller type column ###
    df = pd.get_dummies(df, columns=["Traveller_Type"], prefix="Traveller")

    ### do a sparse encoding of the verified column ###
    df = pd.get_dummies(df, columns=["Verified"], prefix="Verified", drop_first=True)

    ### use the frequency encoding for the passenger name ###
    freq_encoding: pd.Series = df["Passanger_Name"].value_counts(normalize=True)
    df["Passanger_Name"] = df["Passanger_Name"].map(freq_encoding)

    ### run the sentiment analysis on the review content ###
    tqdm.pandas(desc="Computing sentiment scores")
    sentiment_scores: pd.DataFrame = (
        df["Review_content"]
        .progress_apply(
            lambda x: get_sentiment_scores(x)[0] if isinstance(x, str) else {}
        )
        .apply(pd.Series)
    )
    df = (
        pd.concat([df, sentiment_scores], axis=1)
        .drop(columns="Review_content")
        .rename(
            columns={
                0: "Sentiment_Compound",
            }
        )
    )

    ### drop the review title column ###
    """
    we do so as running the sentiment analysis on it resulted in a very sparse distribution
    """
    df.drop(columns=["Review_title"], inplace=True)

    ### make sure to drop any columns that are no longger needed ###
    object_cols = df.select_dtypes(include=["object"]).columns
    df.drop(columns=object_cols, inplace=True)

    return df


def process_spatial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process flight-related data, by doing the following:
        - ...
    Args:
        df, pd.DataFrame: The input DataFrame containing flight data.
    Returns:
        df, pd.DataFrame: The processed DataFrame.
    """
    ### copy the df to avoid modifying the original ###
    df = df.copy()

    ### drop some columns ###
    """
    1) Flying_Date: The vast majority of the dates are missing, and
       cannot be extracted from other columns.
    2) Route: The route can be inferred from the start and
       end locations and any layovers, so it is redundant information.
    """
    df.drop(columns=["Flying_Date", "Route"], inplace=True)

    return df


def main() -> int:
    """"""
    ### init some stuff ###
    input_path: str = "./dbs/raw/db.csv"

    ### load the data ###
    df: pd.DataFrame = load_data(input_path)

    ### ~~~ BETA ~~~ ###
    df = df[:]

    ### process passenger data ###
    # df_passenger: pd.DataFrame = process_passenger_data(df[COLS_PASSENGER])

    ### process flight data ###
    df_spatial: pd.DataFrame = process_spatial_data(df[COLS_SPATIAL])

    ### ~~~ BETA ~~~ ###
    """
    Notes: start location and end location AND LAYOVER can NOT be missing if the route is present and vs
    therefore any string manipulation on the route column should be unnecessary
    Finally that means that the routes is irrelevant and should be dropped
    Another note i know for fact that if a lat was given ill always have the long for it.
    The missing values count for the various columns are as follows:
    1) Start_Address: 1.67%
    2) End_Address: 2.76%
    3) Start_Location: 21.42%
    4) End_Location: 21.42%
    5) Start_Latitude: 1.67%
    6) End_Latitude: 2.76%
    """
    cols = ["Start_Address", "End_Address"]
    alt_cols = ["Start_Location", "End_Location"]
    alt2_cols = ["Start_Latitude", "End_Latitude"]
    print(df_spatial[cols].isna().sum() / len(df_spatial) * 100)
    print(df_spatial[alt_cols].isna().sum() / len(df_spatial) * 100)
    print(df_spatial[alt2_cols].isna().sum() / len(df_spatial) * 100)
    return 0


if __name__ == "__main__":
    exit(main())

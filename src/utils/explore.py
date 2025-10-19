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

    ### drop the flying date column and layover route ###
    """
    As around 73% and 86% of the data respectively is missing
    """
    # df.drop(columns=["Flying_Date", "Layover_Route"], inplace=True)

    ### split the route on the str `to` ###
    """
    To do so we need to do the following:
        1) strip any leading/trailing whitespace
        2) get all masks:
            1) get all routes that contain ` to ` (with spaces)
            2) get all routes that DO NOT contain ` to ` (with spaces)
            3) get all routes with NaNs
        3) for routes that contain ` to `, split on it and keep both parts
            1) for the first part, assign it to a new column `Route_From`
               and Rote_To respectively
            2) for the second part, try to split with the middle space; if
               that fails, assign NaN to Route_From and Route_To
            3) for the final part do the same as step 3.2 in the false case
    """
    ## step 1 ##
    df.Route = df.Route.str.strip()

    ## step 2 ##
    # step 2.1 #
    mask_contains_to: pd.Series = df.Route.str.contains(" to ", na=False)
    # step 2.2 #
    mask_not_contains_to: pd.Series = ~df.Route.str.contains(" to ", na=True)
    # step 2.3 #
    mask_nan: pd.Series = df.Route.isna()

    ## step 3 ##
    # step 3.1 #
    df.loc[mask_contains_to, "Route_From"] = (
        df.loc[mask_contains_to, "Route"].str.split(" to ").str[0].str.strip()
    )
    df.loc[mask_contains_to, "Route_To"] = (
        df.loc[mask_contains_to, "Route"].str.split(" to ").str[1].str.strip()
    )
    # step 3.2 #
    # tmp = df[mask_not_contains_to & df.Route.str.count(" ") == 1]
    # use the abouve code to update the mask and the apply the operation
    updated_mask_not_contains_to: pd.Series = (
        mask_not_contains_to & df.Route.str.count(" ") == 1
    )
    df.loc[updated_mask_not_contains_to, "Route_From"] = (
        df.loc[updated_mask_not_contains_to, "Route"].str.split(" ").str[0].str.strip()
    )
    df.loc[updated_mask_not_contains_to, "Route_To"] = (
        df.loc[updated_mask_not_contains_to, "Route"].str.split(" ").str[1].str.strip()
    )
    # step 3.3 #
    df.loc[mask_nan | (df.Route.str.count(" ") != 1), "Route_From"] = pd.NA
    df.loc[mask_nan | (df.Route.str.count(" ") != 1), "Route_To"] = pd.NA

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
    """
    tmp = df[COLS_SPATIAL]
    return 0


if __name__ == "__main__":
    exit(main())

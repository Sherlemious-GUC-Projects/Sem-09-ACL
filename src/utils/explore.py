### ~~~ GLOBAL IMPORTS ~~~ ###
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

### ~~~ LOCAL IMPORTS ~~~ ###
from explore_util import (
    load_data,
    get_sentiment_scores,
    load_airports_csv,
    build_spatial_backend,
    map_dataframe_coords_to_airport,
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
    df["Class_Encoded"] = df["Class"].map(mapper)
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


def process_spatial_data(
    df: pd.DataFrame,
    airport_path: str = "./dbs/raw/airports.csv",
) -> pd.DataFrame:
    """
    Process flight-related data, by doing the following:
        - Dropping columns
        - Mapping start and end locations to nearest airports
        - Converting layover information to a binary indicator
        - Encoding categorical variables (continent, country, airport codes)
          * using a hierarchical encoding scheme to avoid information loss
          * Do a quick sanity check to insure there is a one-one mapping with the codes
        - Dropping any remaining object-type columns.
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

    ### extract the airport info from the start and end locations ###
    ## 1. load the airports data ##
    df_airports: pd.DataFrame = load_airports_csv(airport_path)
    ## 2. build the spatial backend ##
    spatial_backend = build_spatial_backend(df_airports)
    ## 3. map the start location to the nearest airport ##
    df = map_dataframe_coords_to_airport(
        df=df,
        lat_col="Start_Latitude",
        lon_col="Start_Longitude",
        backend=spatial_backend,
        prefix="Start_",
    )
    ## 4. map the end location to the nearest airport ##
    df = map_dataframe_coords_to_airport(
        df=df,
        lat_col="End_Latitude",
        lon_col="End_Longitude",
        backend=spatial_backend,
        prefix="End_",
    )
    ## 5. drop the remaining spatial columns ##
    df.drop(
        columns=[
            "Start_Location",
            "End_Location",
            "Start_Address",
            "End_Address",
            "Start_Latitude",
            "Start_Longitude",
            "End_Latitude",
            "End_Longitude",
            "Start_Dist_km",
            "End_Dist_km",
            "Start_Name",  # they are redundant with the codes
            "End_Name",  # they are redundant with the codes
        ],
        inplace=True,
    )

    ### convert the Layover column to a binary indicator ###
    df["Has_Layover"] = df.Layover_Route.notna().astype(int)
    df.drop(columns=["Layover_Route"], inplace=True)

    ### Encode the Airline column ###
    ## 1. do a sparse encoding of the continent columns ##
    df = pd.get_dummies(
        df, columns=["Start_Continent"], prefix="Start_Cont", drop_first=True, dtype=int
    )
    df = pd.get_dummies(
        df, columns=["End_Continent"], prefix="End_Cont", drop_first=True, dtype=int
    )
    ## 2. do a frequency encoding of the country ##
    df["Start_Country_freq"] = df["Start_Country"].map(
        df["Start_Country"].value_counts()
    )
    df["End_Country_freq"] = df["End_Country"].map(df["End_Country"].value_counts())
    ## 3. do a frequency encoding of the airport codes ##
    df["Start_Code_freq"] = df["Start_Code"].map(df["Start_Code"].value_counts())
    df["End_Code_freq"] = df["End_Code"].map(df["End_Code"].value_counts())

    ### sanity check ###
    ## 1. get the columns of interest ##
    Start_cols_of_interest = [c for c in df.columns if c.startswith("Start_Cont_")] + [
        "Start_Country_freq",
        "Start_Code_freq",
    ]
    End_cols_of_interest = [c for c in df.columns if c.startswith("End_Cont_")] + [
        "End_Country_freq",
        "End_Code_freq",
    ]
    ## 2. check for one-to-one mapping ##
    Start_is_one_to_one = (
        df.groupby("Start_Code")[Start_cols_of_interest]
        .nunique()
        .apply(lambda s: (s <= 1).all(), axis=1)
        .all()
    )
    End_is_one_to_one = (
        df.groupby("End_Code")[End_cols_of_interest]
        .nunique()
        .apply(lambda s: (s <= 1).all(), axis=1)
        .all()
    )
    ## 3. assert the one-to-one mapping ##
    assert (
        Start_is_one_to_one
    ), "Start_Code encoding is not one-to-one with the Start code leading to information loss"
    assert (
        End_is_one_to_one
    ), "End_Code encoding is not one-to-one with the End code leading to information loss"

    ### drop any remaining object columns ###
    object_cols = df.select_dtypes(include=["object"]).columns
    df.drop(columns=object_cols, inplace=True)

    return df


def main() -> int:
    """"""
    ### init some stuff ###
    input_path: str = "./dbs/raw/db.csv"

    ### load the data ###
    df: pd.DataFrame = load_data(input_path)

    ### drop all records with NaN in the coords columns ###
    df.dropna(
        subset=[
            "Start_Latitude",
            "Start_Longitude",
            "End_Latitude",
            "End_Longitude",
        ],
        inplace=True,
    )

    ### ~~~ BETA ~~~ ###
    df = df[:3]

    ### process passenger data ###
    df_passenger: pd.DataFrame = process_passenger_data(df[COLS_PASSENGER])

    ### process flight data ###
    df_spatial: pd.DataFrame = process_spatial_data(df[COLS_SPATIAL])

    ### concatenate the dataframes ###
    df_final: pd.DataFrame = pd.concat([df_passenger, df_spatial], axis=1)
    df_final = df_final.astype(float)

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
    return 0


if __name__ == "__main__":
    exit(main())

### ~~~ GLOBAL IMPORTS ~~~ ###
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

### ~~~ LOCAL IMPORTS ~~~ ###
from explore_util import (
    load_data,
    COLS,
    COLS_PASSENGER,
    COLS_FLIGHT,
    COLS_SPATIAL,
)


def process_passenger_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process passenger-related data. It does the following:
      - Dense encoding of the 'Class' column. (possibly apply an exponential growth encoding later)
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

    ### make sure to drop any columns that are no longger needed ###
    object_cols = df.select_dtypes(include=["object"]).columns
    print(object_cols)

    return df


def main() -> int:
    """"""
    ### init some stuff ###
    input_path: str = "./dbs/raw/db.csv"

    ### load the data ###
    df: pd.DataFrame = load_data(input_path)

    ### process passenger data ###
    df_passenger: pd.DataFrame = process_passenger_data(df[COLS_PASSENGER])

    ### cool stuff ###

    return 0


if __name__ == "__main__":
    exit(main())

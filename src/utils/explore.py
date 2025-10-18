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


def main() -> int:
    """"""
    ### init some stuff ###
    input_path: str = "./dbs/raw/db.csv"

    ### load the data ###
    df: pd.DataFrame = load_data(input_path)

    ### ~~~ BETA ~~~ ###
    df = df[:500]

    ### process passenger data ###
    df_passenger: pd.DataFrame = process_passenger_data(df[COLS_PASSENGER])

    ### cool stuff ###
    df_passenger["Sentiment_Compound"].hist(bins=500)
    plt.title("Sentiment Compound Score Distribution")
    plt.xlabel("Sentiment Compound Score")
    plt.ylabel("Frequency")
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())

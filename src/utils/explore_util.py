### ~~~ GLOBAL IMPORTS ~~~ ###
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
    # flight data
    "Flying_Date",
    "Layover_Route",
    "Route",
    # spatial data
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
COLS_FLIGHT = COLS[7:11]
COLS_SPATIAL = COLS[11:]


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


def pct_nan(df: pd.DataFrame, cols: list[str]) -> float:
    """Mean of per-column NaN percentages for the given columns."""
    return df[cols].isna().mean().mean() * 100


def print_nan_overview(df: pd.DataFrame) -> None:
    """Log NaN exploration using your existing print format."""
    print("Nans for passenger data:", pct_nan(df, COLS_PASSENGER))
    # print(df[COLS_PASSENGER].isna().sum() / len(df) * 100)

    print("Nans for flight data:", pct_nan(df, COLS_FLIGHT))
    # print(df[COLS_FLIGHT].isna().sum() / len(df) * 100)

    print("Nans for spatial data:", pct_nan(df, COLS_SPATIAL))
    print(df[COLS_SPATIAL].isna().sum() / len(df) * 100)
    print("-" * 20)


def print_spatial_uniques(df: pd.DataFrame) -> None:
    """Log simple spatial uniqueness stats (your notation preserved)."""
    print(
        "Unique locations:",
        df["Start_Location"].nunique() + df["End_Location"].nunique(),
    )
    print(
        "Unique  latitudes:",
        df["Start_Latitude"].nunique() + df["End_Latitude"].nunique(),
    )


def to_str(lat, long) -> str:
    """String hash for (lat, long) as you defined it."""
    return f"{lat:.6f}_{long:.6f}"


def add_latlong_strings(df: pd.DataFrame) -> None:
    """Add Start/End_LatLong_Str columns in-place (keeps your column names)."""
    df["Start_LatLong_Str"] = df.apply(
        lambda row: to_str(row["Start_Latitude"], row["Start_Longitude"]), axis=1
    )
    df["End_LatLong_Str"] = df.apply(
        lambda row: to_str(row["End_Latitude"], row["End_Longitude"]), axis=1
    )


def build_lat_long_loc_map(df: pd.DataFrame) -> dict[str, list]:
    """
    Construct the mapping from LatLong string to associated locations
    from both Start_ and End_ sides, de-duped while preserving order.
    """
    lat_long_index = (
        pd.concat([df["Start_LatLong_Str"], df["End_LatLong_Str"]]).unique().tolist()
    )

    lat_long_loc_map: dict = {i: [] for i in lat_long_index}
    for index in lat_long_index:
        locs_start = (
            df.loc[df["Start_LatLong_Str"] == index, "Start_Location"].dropna().tolist()
        )
        locs_end = (
            df.loc[df["End_LatLong_Str"] == index, "End_Location"].dropna().tolist()
        )

        # merge start+end, de-dupe, preserve first-seen order
        seen = set()
        merged: list = []
        for x in locs_start + locs_end:
            if x not in seen:
                seen.add(x)
                merged.append(x)

        lat_long_loc_map[index] = merged

    return lat_long_loc_map


def main() -> int:
    """"""
    ### init some stuff ###
    input_path: str = "./dbs/raw/db.csv"

    ### load the data ###
    df: pd.DataFrame = load_data(input_path)

    return 0


if __name__ == "__main__":
    exit(main())

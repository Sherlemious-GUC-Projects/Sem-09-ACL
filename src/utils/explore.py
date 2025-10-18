### ~~~ GLOBAL IMPORTS ~~~ ###
from matplotlib import pyplot as plt
import pandas as pd
import pathlib
import json

### ~~~ LOCAL IMPORTS ~~~ ###
from explore_util import (
    load_data,
    COLS,
    COLS_PASSENGER,
    COLS_FLIGHT,
    COLS_SPATIAL,
)


def main() -> int:
    """"""
    ### init some stuff ###
    input_path: str = "./dbs/raw/db.csv"

    ### load the data ###
    df: pd.DataFrame = load_data(input_path)

    return 0


if __name__ == "__main__":
    exit(main())

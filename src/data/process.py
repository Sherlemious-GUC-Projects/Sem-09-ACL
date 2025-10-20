### ~~~ GLOBAL IMPORTS ~~~ ###
from sklearn.model_selection import train_test_split
from typing import TypeAlias
import pandas as pd
import numpy as np

### ~~~ LOCAL IMPORTS ~~~ ###
from util import (
    load_data,
)

### ~~~ CUSTOM TYPES ~~~ ###
tensor_t: TypeAlias = np.ndarray

### ~~~ STATE DEFINITIONS ~~~ ###
# None


def main() -> int:
    """"""
    ### init some stuff ###
    input_path: str = "./dbs/interm/db.csv"
    output_path: str = "./dbs/cooked/data.npz"

    ### load the data ###
    df: pd.DataFrame = load_data(input_path)

    ### get the X and y ###
    X: tensor_t = df.drop(columns=["Rating"]).to_numpy()
    y: tensor_t = df["Rating"].to_numpy()

    ### split the data ###
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    ### save the data as one big npz file ###
    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_temp=X_temp,
        y_temp=y_temp,
    )

    return 0


if __name__ == "__main__":
    exit(main())

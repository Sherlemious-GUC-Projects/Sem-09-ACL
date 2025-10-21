### ~~~ GLOBAL IMPORTS ~~~ ###
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from typing import TypeAlias, List
import pandas as pd
import numpy as np

### ~~~ LOCAL IMPORTS ~~~ ###
from util import (
    load_data,
)

### ~~~ CUSTOM TYPES ~~~ ###
tensor_t: TypeAlias = np.ndarray
list_str_t: TypeAlias = List[str]

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
    X_df: pd.DataFrame = df.drop(columns=["Rating"])
    y: tensor_t = df["Rating"].to_numpy()

    ### identify column types for scaling ###
    binary_cols: list_str_t = [
        col
        for col in X_df.columns
        if col.startswith("Traveller_")
        or col.startswith("Verified_")
        or col.startswith("Start_Cont_")
        or col.startswith("End_Cont_")
        or col == "Has_Layover"
    ]
    scalable_cols: list_str_t = [col for col in X_df.columns if col not in binary_cols]

    ### define the preprocessor ###
    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), scalable_cols),
            ("passthrough", "passthrough", binary_cols),
        ],
    )

    ### get the column names as it might be useful later ###
    y_col_name: list_str_t = ["Rating"]

    ### split the data ###
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )

    ### fit and transform the data ###
    X_train: tensor_t = preprocessor.fit_transform(X_train_df)
    X_test: tensor_t = preprocessor.transform(X_test_df)

    ### get the new column order from the transformer ###
    X_col_names: list_str_t = preprocessor.get_feature_names_out()

    ### save the data as one big npz file ###
    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_col_names=X_col_names,
        y_col_name=y_col_name,
    )

    return 0


if __name__ == "__main__":
    exit(main())

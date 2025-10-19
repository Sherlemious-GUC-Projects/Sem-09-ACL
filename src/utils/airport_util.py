### ~~~ GLOBAL IMPORTS ~~~ ###
from __future__ import annotations
from typing import Iterable, Optional, Tuple, TypedDict
from sklearn.neighbors import BallTree
import pandas as pd
import numpy as np


### ~~~ CUSTOM TYPES ~~~ ###
class Backend(TypedDict):
    tree: BallTree
    lat_rad: np.ndarray
    lon_rad: np.ndarray
    codes: np.ndarray
    names: np.ndarray
    continents: np.ndarray
    countries: np.ndarray


### ~~~ CONSTANTS ~~~ ###
EARTH_RADIUS_KM: float = 6371.0088
K_NEIGHBORS: int = 1


def _to_radians(x: np.ndarray) -> np.ndarray:
    """Convert degrees → radians (vectorized)."""
    return np.radians(x.astype(float))


def load_airports_csv(
    path: str,
    include_types: Tuple[str, ...] = (
        "large_airport",
        "medium_airport",
    ),
) -> pd.DataFrame:
    """
    Load the OurAirports airports.csv and do minimal cleaning.
        - Reads the CSV.
        - Filters rows by 'type' ∈ include_types.
        - Drops rows with missing lat/lon.
        - Resolves a canonical 'Code' column: prefer IATA, else ICAO 'ident'.
        - Keeps only the columns we need, and adds radians for lat/lon.
    Args:
        path, str: Path to OurAirports airports.csv.
        include_types, tuple[str, ...]: Airport types to retain.
    Returns:
        df, pd.DataFrame: Cleaned airports with columns:
            ['Code','Name','Type','Latitude','Longitude','Lat_rad','Lon_rad']
    """
    ### read and copy ###
    df: pd.DataFrame = pd.read_csv(path).copy()

    ### filter on type + valid coords ###
    df = df[df["type"].isin(include_types)]
    df = df[(~df["latitude_deg"].isna()) & (~df["longitude_deg"].isna())]

    ### resolve canonical code (IATA else ICAO) ###
    code: pd.Series = np.where(
        df["iata_code"].notna() & (df["iata_code"].astype(str).str.len() > 0),
        df["iata_code"].astype(str),
        df["ident"].astype(str),
    )  # type: ignore[assignment]

    ### select + rename to your style ###
    # print(df[["continent", "iso_country"]])
    # exit()
    out: pd.DataFrame = (
        df.assign(Code=code)
        .rename(
            columns={
                "name": "Name",
                "type": "Type",
                "latitude_deg": "Latitude",
                "longitude_deg": "Longitude",
            }
        )[["Code", "Name", "Type", "Latitude", "Longitude", "continent", "iso_country"]]
        .copy()
    )

    ### radians (pure add) ###
    out["Lat_rad"] = _to_radians(out["Latitude"].to_numpy())
    out["Lon_rad"] = _to_radians(out["Longitude"].to_numpy())

    return out


def build_spatial_backend(
    airports: pd.DataFrame,
) -> Backend:
    """
    Prepare a pure, serializable description of the spatial backend.
        - If scikit-learn is available: use BallTree with haversine metric.
        - Else if SciPy is available: use cKDTree on unit sphere (XYZ).
        - Else: fallback = no tree (we’ll do brute-force haversine).
    Args:
        airports, pd.DataFrame: Output of load_airports_csv(...).
    Returns:
        backend, Backend: A functional backend descriptor with:
            {
              'tree': object-or-None,
              'lat_rad': np.ndarray,
              'lon_rad': np.ndarray,
              'codes': np.ndarray,
              'names': np.ndarray
              'continents': np.ndarray,
              'countries': np.ndarray,
            }
    """
    ### extract arrays (no mutation) ###
    lat_rad: np.ndarray = airports["Lat_rad"].to_numpy()
    lon_rad: np.ndarray = airports["Lon_rad"].to_numpy()
    codes: np.ndarray = airports["Code"].astype(str).to_numpy()
    names: np.ndarray = airports["Name"].astype(str).to_numpy()
    continents: np.ndarray = airports["continent"].astype(str).to_numpy()
    countries: np.ndarray = airports["iso_country"].astype(str).to_numpy()

    ### BallTree backend ###
    tree: BallTree = BallTree(np.c_[lat_rad, lon_rad], metric="haversine")

    ### construct the backend object ###
    backend: Backend = {
        "tree": tree,
        "lat_rad": lat_rad,
        "lon_rad": lon_rad,
        "codes": codes,
        "names": names,
        "continents": continents,
        "countries": countries,
    }

    return backend


def nearest_airport_batch(
    coords: Iterable[Tuple[float, float]],
    backend: Backend,
    max_km: Optional[float] = None,
) -> pd.DataFrame:
    """
    Vectorized batch nearest lookup.
        - Accepts an iterable of (lat, lon) in degrees.
        - Returns a DataFrame aligned to input order.
        - Applies max_km if provided (rows with no match become NaN/None).
    Args:
        coords, Iterable[(float, float)]: Sequence of query coordinates.
        backend, Backend: Output of build_spatial_backend(...).
        max_km, float|None: Optional maximum distance in km.
    Returns:
        df, pd.DataFrame: Columns [
            'Query_Lat',
            'Query_Lon',
            'Code',
            'Name',
            'Distance_km',
            'Continents',
            'Countries',
            'Index'
        ]
    """
    ### materialize inputs (no mutation) ###
    coords_arr: np.ndarray = np.asarray(list(coords), dtype=float)
    if coords_arr.size == 0:
        return pd.DataFrame(
            columns=["Query_Lat", "Query_Lon", "Code", "Name", "Distance_km", "Index"]
        )

    ### prepare queries in radians ###
    q_lat = coords_arr[:, 0]
    q_lon = coords_arr[:, 1]
    q_lat_rad = _to_radians(q_lat)
    q_lon_rad = _to_radians(q_lon)

    ### fetch backend data ###
    codes: np.ndarray = backend["codes"]
    names: np.ndarray = backend["names"]
    continents: np.ndarray = backend["continents"]
    countries: np.ndarray = backend["countries"]

    ### query the tree by concatenated radians ###
    dist_rad, idx = backend["tree"].query(np.c_[q_lat_rad, q_lon_rad], k=K_NEIGHBORS)
    j = idx[:, 0].astype(int)

    ### convert to km ###
    d_km = (dist_rad[:, 0] * EARTH_RADIUS_KM).astype(float)

    ### assemble result (apply radius if set) ###
    code_out = codes[j].astype(str)
    name_out = names[j].astype(str)
    continent_out = continents[j].astype(str)
    country_out = countries[j].astype(str)

    ### apply max_km if set ###
    if max_km is not None:
        mask = d_km <= max_km
        code_out = np.where(mask, code_out, None)  # type: ignore[assignment]
        name_out = np.where(mask, name_out, None)  # type: ignore[assignment]
        continent_out = np.where(mask, continent_out, None)  # type: ignore[assignment]
        country_out = np.where(mask, country_out, None)  # type: ignore[assignment]
        d_km = np.where(mask, d_km, np.nan)
        j = np.where(mask, j, -1)

    return pd.DataFrame(
        {
            "Query_Lat": q_lat,
            "Query_Lon": q_lon,
            "Code": code_out,
            "Name": name_out,
            "Continent": continent_out,
            "Country": country_out,
            "Distance_km": d_km,
            "Index": j,
        }
    )


def map_dataframe_coords_to_airport(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    backend: Backend,
    prefix: str = "Nearest_",
    max_km: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper to keep your pandas pipeline clean.
        - Does NOT mutate the input DataFrame.
        - Returns a new DataFrame with ['Nearest_Code','Nearest_Name','Nearest_Dist_km'] added.
    Args:
        df, pd.DataFrame: Input table with latitude/longitude columns.
        lat_col, str: Name of the latitude column in df (degrees).
        lon_col, str: Name of the longitude column in df (degrees).
        backend, dict: Output of build_spatial_backend(...).
        prefix, str: Prefix for the new columns.
        max_km, float|None: Optional maximum distance in km.
    Returns:
        df, pd.DataFrame: Copy of df with 3 appended columns.
    """
    ### copy the df to avoid modifying the original ###
    base = df.copy()

    ### compute batch nearest ###
    results = nearest_airport_batch(
        coords=list(zip(base[lat_col].to_list(), base[lon_col].to_list())),
        backend=backend,
        max_km=max_km,
    )

    ### merge columns in your style ###
    out = base.assign(
        **{
            f"{prefix}Code": results["Code"].to_numpy(),
            f"{prefix}Name": results["Name"].to_numpy(),
            f"{prefix}Continent": results["Continent"].to_numpy(),
            f"{prefix}Country": results["Country"].to_numpy(),
            f"{prefix}Dist_km": results["Distance_km"].to_numpy(),
        }
    )

    return out


def test() -> int:
    """"""
    path = "./dbs/raw/airports.csv"
    df_path = "./dbs/raw/db.csv"

    airports = load_airports_csv(path)
    df = pd.read_csv(df_path)
    backend = build_spatial_backend(airports)

    df_mapped = map_dataframe_coords_to_airport(
        df.dropna(subset=["Start_Latitude", "Start_Longitude"]),
        lat_col="Start_Latitude",
        lon_col="Start_Longitude",
        backend=backend,
        prefix="start_",
        max_km=100,
    )
    print(
        df_mapped[
            [
                "Start_Latitude",
                "Start_Longitude",
                "start_Code",
                "start_Name",
                "start_Dist_km",
            ]
        ].head(10)
    )
    return 0


if __name__ == "__main__":
    test()

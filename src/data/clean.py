import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import airportsdata as ad
from tqdm import tqdm

### ~~~ CSV file paths ~~~ ###
AIRLINE_SCRAPPED_REVIEW_CLEANED_PATH = "../../dbs/raw/AirlineScrappedReview_Cleaned.csv"
CUSTOMER_COMMENT_PATH = "../../dbs/raw/Customer_comment.csv"
PASSANGER_BOOKING_DATA_PATH = "../../dbs/raw/Passanger_booking_data.csv"
SURVEY_DATA_INFLIGHT_SATISFACTION_SCORE_PATH = "../../dbs/raw/Survey data_Inflight Satisfaction Score.csv"

### ~~~ Load the dataframes ~~~ ###
df_0: pd.DataFrame = pd.read_csv(AIRLINE_SCRAPPED_REVIEW_CLEANED_PATH)
df_1: pd.DataFrame = pd.read_csv(CUSTOMER_COMMENT_PATH)
df_2: pd.DataFrame = pd.read_csv(PASSANGER_BOOKING_DATA_PATH)
df_3: pd.DataFrame = pd.read_csv(SURVEY_DATA_INFLIGHT_SATISFACTION_SCORE_PATH)

#global variable i think? 
list_of_dfs: list[pd.DataFrame] = [df_0, df_1, df_2, df_3]


### this works for customer comment and customer survey data inflight satisfaction score data
def find_out_missing_aiports_from_airports_data(df: pd.DataFrame) -> list[str]:
    """
    Find out missing airports from the dataframe. // work on the surey data inflight satisfaction score and when ad throws error catch it and continue but put the missing airports in a list and return it. it exists inside origin_station_code and destination_station_code.
    Args:
        df, pd.DataFrame: The dataframe containing the flight data.
    Returns:
        list[str]: The list of missing airports.
    """
    #traverse the dataframe and check if the origin_station_code and destination_station_code exist in the ad dictionary    if not add it to the missing_airports list
    # add a progress bar to the function
    print("Finding out missing airports...")
    missing_airports: list[str] = []
    seen: set[str] = set()
    iata = ad.load('IATA')
    for i in tqdm(range(len(df))):
        origin_val = df["origin_station_code"].iloc[i] if "origin_station_code" in df.columns else None
        dest_val = df["destination_station_code"].iloc[i] if "destination_station_code" in df.columns else None

        origin = str(origin_val).strip() if pd.notna(origin_val) else None
        dest = str(dest_val).strip() if pd.notna(dest_val) else None

        if origin and origin not in iata and origin not in seen:
            missing_airports.append(origin)
            seen.add(origin)
        if dest and dest not in iata and dest not in seen:
            missing_airports.append(dest)
            seen.add(dest)
    return missing_airports

def load_flight_data(list_of_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Load flight data from a list of DataFrames.
    ####
    this function is used to load the flight data from the list of dataframes and combine them before further cleaning and processing under one column [routes]
    ####
    Args:
        df, list[pd.DataFrame]: The list of DataFrames containing flight data.
        note : the list needs to contain the following dataframes in order
        - AirlineScrappedReview_Cleaned.csv [Start_Location, End_Location]
        - Customer_comment.csv[origin_station_code, destination_station_code]
        - Passanger_booking_data.csv[route]
        - Survey data_Inflight Satisfaction Score.csv[origin_station_code, destination_station_code]
    Returns:
        df, pd.DataFrame: The loaded flight data as a pandas DataFrame.
    """
    # traverse the list of dataframes and combine them into one dataframe
    # initialize a new dataframe to store the combined data
    combined_df = pd.DataFrame(columns=['routes'])

    for df in list_of_dfs:
        if "Start_Location" in df.columns and "End_Location" in df.columns:
            r = df["Start_Location"].astype(str).str.strip() + " -> " + df["End_Location"].astype(str).str.strip()

        elif "origin_station_code" in df.columns and "destination_station_code" in df.columns:
            r = df["origin_station_code"].astype(str).str.strip() + " -> " + df["destination_station_code"].astype(str).str.strip()

        elif "route" in df.columns:
            r = df["route"].astype(str).str.strip()
        else:
            continue

        combined_df = pd.concat([combined_df, pd.DataFrame({"routes": r})], ignore_index=True)

    # optional cleanup
    combined_df = combined_df.dropna(subset=["routes"])

    return combined_df

def main() -> int:
    """
    Main function to load flight data.
    """

    #test the find_out_missing_aiports function
    missing_airports = find_out_missing_aiports_from_airports_data(df_3)
    print(missing_airports)
    
    #load the flight data
    combined_df = load_flight_data(list_of_dfs)
    #show me the first 5 rows of the combined dataframe
    print(combined_df.head(5))
    #show me the last 5 rows of the combined dataframe
    print(combined_df.tail(5))
    #show me the shape of the combined dataframe
    print(combined_df.shape)
    #show me the columns of the combined dataframe
    print(combined_df.columns)
    
    #load the airport data
    iata = ad.load('IATA')
    info = iata['ICN']
    print(info['city'], info['country'], info['lat'], info['lon'])
    return 0;

if __name__ == "__main__":
    main()
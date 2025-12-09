# Airline Knowledge Graph Project

## Overview

This project transforms airline survey data into a Neo4j Knowledge Graph (KG) to enable complex analysis of passenger experiences, flight operations, and travel patterns. It provides a complete pipeline from raw CSV data extraction to graph construction and analytical querying.

## Architecture

The system follows a standard ETL (Extract, Transform, Load) pattern implemented in Python:

1.  **Extract:** Reads raw survey data from `dbs/Airline_surveys_sample.csv`.
2.  **Transform:** `src/construct.py` cleanses the data and models it into graph entities (Nodes) and relationships (Edges).
3.  **Load:** `src/load.py` efficiently batches these entities into a Neo4j database using Cypher `UNWIND` operations.
4.  **Analyze:** `src/query.py` executes specific analytical questions against the graph.
5.  **Validate:** `src/main.py` compares the live query results against pre-calculated expected answers stored in `dbs/query_*.json`.

## Knowledge Graph Specification

### Node Labels & Properties

The graph schema is strictly typed with the following nodes:

- **`Passenger`**
  - `record_locator` (String, Unique ID): The passenger's unique booking reference.
  - `loyalty_program_level` (String): Status level (e.g., "Silver", "Gold").
  - `generation` (String): Demographic generation (e.g., "Gen X", "Baby Boomers").

- **`Journey`**
  - `feedback_ID` (String, Unique ID): Unique identifier for the specific trip/survey response.
  - `food_satisfaction_score` (Integer): Rating of food quality.
  - `arrival_delay_minutes` (Integer): Minutes delayed at arrival (can be negative for early arrivals).
  - `actual_flown_miles` (Integer): Distance of the journey.
  - `number_of_legs` (Integer): Count of flight segments in the journey.
  - `passenger_class` (String): Cabin class (e.g., "Economy", "Business").

- **`Flight`**
  - `flight_number` (String): The flight identifier.
  - `fleet_type_description` (String): The aircraft model family (e.g., "737 Max").
  - _Constraint:_ Uniqueness is based on the combination of `flight_number` AND `fleet_type_description`.

- **`Airport`**
  - `station_code` (String, Unique ID): IATA 3-letter airport code (e.g., "LHR", "JFK").

### Relationships

- `(:Passenger)-[:TOOK]->(:Journey)`: Links a passenger to the specific trip they experienced.
- `(:Journey)-[:ON]->(:Flight)`: Connects the journey experience to the specific flight operation.
- `(:Flight)-[:DEPARTS_FROM]->(:Airport)`: Origin of the flight.
- `(:Flight)-[:ARRIVES_AT]->(:Airport)`: Destination of the flight.

## Analytical Queries

The project implements 6 specific queries to demonstrate the graph's analytical capabilities. These are defined in `src/query.py`:

1.  **Top Busiest Routes:**
    - _Goal:_ Identify the 5 most frequent airport pairs.
    - _Logic:_ Counts `(Flight)` connections between `(Airport)` nodes.

2.  **Flights with Most Feedback:**
    - _Goal:_ Find the top 10 flights receiving the most survey responses.
    - _Logic:_ Counts `(Journey)` paths connected to each `(Flight)`.

3.  **Food Satisfaction by Generation (Multi-leg):**
    - _Goal:_ Analyze if older/younger generations prefer food on longer (multi-leg) trips.
    - _Logic:_ Filters for `number_of_legs > 1`, groups by `generation`, and averages `food_satisfaction_score`.

4.  **Best On-Time Performance:**
    - _Goal:_ Find the top 10 flights with the lowest average arrival delay.
    - _Logic:_ Averages `arrival_delay_minutes` from linked `(Journey)` nodes.

5.  **Distance vs. Loyalty:**
    - _Goal:_ See if higher loyalty tiers correlate with longer distances flown.
    - _Logic:_ Groups by `loyalty_program_level` and averages `actual_flown_miles`.

6.  **Complex Satisfaction Index:**
    - _Goal:_ Calculate a custom "Satisfied Passenger" count based on a weighted formula.
    - _Formula:_
      - `Score = (0.5 * Food) + (0.35 * Delay_Score) + (0.1 * Legs_Score) + (0.05 * Miles_Score)`
      - _Sub-scores_ (Delay, Legs, Miles) are normalized on a 0-5 scale.
      - Passengers with `Score > 3` are counted as "Satisfied".

## Recreation & Usage Guide

### 1. Configuration

Create a `config.txt` file in the project root to connect to your Neo4j instance:

```properties
URI=bolt://localhost:7687
USERNAME=neo4j
PASSWORD=your_secure_password
```

### 2. Loading the Database

**Important:** By default, the `main.py` script _only runs queries_ and assumes the data is already loaded. To populate the database for the first time (or reload it):

1.  Open `src/main.py`.
2.  Uncomment the `loader()` function call inside `main()`:
    ```python
    def main() -> int:
        ### load data ###
        loader()  # <--- Remove the # to enable loading
        ...
    ```
3.  Run the script:
    ```bash
    python src/main.py
    ```
4.  (Optional) Re-comment the line to prevent reloading on subsequent runs.

### 3. Running Analysis

Once the data is loaded, simply run:

```bash
python src/main.py
```

This will:

1.  Connect to Neo4j.
2.  Execute Queries 1-6.
3.  Compare the results against the "Truth" files in `dbs/query_*.json`.
4.  Print "match" or mismatch details for each query.

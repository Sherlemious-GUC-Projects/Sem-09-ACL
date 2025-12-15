### ~~~ GLOBAL IMPORTS ~~~ ###
import argparse
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import chromadb
import numpy as np
import ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

### ~~~ LOCAL IMPORTS ~~~ ###
from src.retrieval.util import load_data
from src.utils.constant import COLLECTION_NAME, CSV_PATH, USE_OLLAMA, VECTOR_DB_PATH
from src.utils.types import ContextChunk, Entity, ProcessedQuery, RetrievalSource

### ~~~ CUSTOM TYPES ~~~ ###
client_t: TypeAlias = chromadb.api.client.Client
tensor_t: TypeAlias = np.ndarray


### ~~~ STATE DEFINITIONS ~~~ ###
class Modes(Enum):
    LOAD = "load"
    DROP = "drop"
    QUERY = "query"


@dataclass
class VectorDb:
    client: client_t
    db_name: str


### ~~~ FUNCTION DEFINITIONS ~~~ ###


def get_client(path: str = VECTOR_DB_PATH) -> client_t:
    """
    Initializes and returns the ChromaDB persistent client.

    Args:
        path: Path to the persistence directory.

    Returns:
        ChromaDB Client instance.
    """
    return chromadb.PersistentClient(path=path)


def embed(
    text: Union[str, List[str]], do_ollama: bool = USE_OLLAMA
) -> Union[tensor_t, List[tensor_t]]:
    """
    Embed the chunks using SentenceTransformer or Ollama.

    Args:
        text: The input text or list of texts to be embedded.
        do_ollama: Whether to use Ollama for embeddings.

    Returns:
        Embeddings as a numpy array or list of arrays.
    """
    model_name: str = (
        "chroma/all-minilm-l6-v2-f32:latest" if do_ollama else "all-MiniLM-L6-v2"
    )

    if do_ollama:
        # Ollama handling
        if isinstance(text, str):
            response = ollama.embeddings(model=model_name, prompt=text)
            return np.array(response["embedding"])
        else:
            # Batch handling for Ollama (looping needed if API doesn't support batch list)
            embeddings = []
            for t in text:
                response = ollama.embeddings(model=model_name, prompt=t)
                embeddings.append(np.array(response["embedding"]))
            return embeddings
    else:
        # SentenceTransformer handling
        model = SentenceTransformer(model_name)
        embeddings = model.encode(text, show_progress_bar=False)
        return np.array(embeddings)


def _format_record(row: pd.Series) -> str:
    """
    Formats a CSV row into the target sentence structure.

    Template:
    "Flight {flight_number} operates from {origin_station_code} to {destination_station_code} as a {number_of_legs}-leg journey using {fleet_type_description} aircraft.
    The route covers {actual_flown_miles} miles and serves passengers in {passenger_class} class.
    Typical arrival delay is {arrival_delay_minutes} minutes.
    Passenger feedback includes a food satisfaction score of {food_satisfaction_score}/5, submitted under feedback ID {feedback_ID}.
    This record reflects a {generation} passenger with {loyalty_program_level} loyalty status."

    Args:
        row: A pandas Series representing a single row from the dataframe.

    Returns:
        A formatted string description of the passenger's experience.
    """
    record = {
        "flight_number": row.get("flight_number", "Unknown"),
        "origin_station_code": row.get("origin_station_code", "Unknown"),
        "destination_station_code": row.get("destination_station_code", "Unknown"),
        "number_of_legs": row.get("number_of_legs", "Unknown"),
        "fleet_type_description": row.get("fleet_type_description", "Unknown"),
        "actual_flown_miles": row.get("actual_flown_miles", 0),
        "passenger_class": row.get("passenger_class", "Unknown"),
        "arrival_delay_minutes": row.get("arrival_delay_minutes", 0),
        "food_satisfaction_score": row.get("food_satisfaction_score", 0),
        "feedback_ID": row.get("feedback_ID", "Unknown"),
        "generation": row.get("generation", "Unknown"),
        "loyalty_program_level": row.get("loyalty_program_level", "Unknown"),
    }
    return (
        f"Flight {record['flight_number']} operates from {record['origin_station_code']} "
        f"to {record['destination_station_code']} as a {record['number_of_legs']}-leg journey "
        f"using {record['fleet_type_description']} aircraft. "
        f"The route covers {record['actual_flown_miles']} miles and serves passengers in "
        f"{record['passenger_class']} class. "
        f"Typical arrival delay is {record['arrival_delay_minutes']} minutes. "
        f"Passenger feedback includes a food satisfaction score of {record['food_satisfaction_score']}/5, "
        f"submitted under feedback ID {record['feedback_ID']}. "
        f"This record reflects a {record['generation']} passenger with "
        f"{record['loyalty_program_level']} loyalty status."
    )


def load_vector_db(
    vector_db: VectorDb, csv_path: str = CSV_PATH, do_ollama: bool = USE_OLLAMA
) -> int:
    """
    Loads up the embeddings into a vector database from the CSV data.
    Get or create a collection based on whether it exists or not.

    Args:
        vector_db: The dataclass with all the info needed to get the collection.
        csv_path: The path to the CSV data.
        do_ollama: Whether to use Ollama for embeddings.

    Returns:
        0 for success.
    """
    try:
        df = load_data(csv_path)
    except FileNotFoundError:
        print(f"WARNING: CSV not found at {csv_path}. Skipping ingestion.")
        return 1

    print(f"Loading data from {csv_path}...")
    
    # Prepare data
    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        text = _format_record(row)
        chunks.append(text)

        metadata = {
            "id": str(row.get("feedback_ID", row.get("record_locator", "unknown"))),
            "flight_number": str(row.get("flight_number", "")),
            "origin": str(row.get("origin_station_code", "")),
            "destination": str(row.get("destination_station_code", "")),
            "date": str(row.get("flight_date", "")),
            "loyalty": str(row.get("loyalty_program_level", "")),
            "class": str(row.get("passenger_class", "")),
        }
        metadatas.append(metadata)

    # Generate IDs
    # Include index to ensure uniqueness for identical texts
    keys: List[str] = [
        hashlib.md5(f"{c}_{i}".encode()).hexdigest() for i, c in enumerate(chunks)
    ]

    # Embed (using batching or loop depending on impl)
    print("Generating embeddings...")
    embeddings_list = []
    
    # We batch process or loop with tqdm here for visibility
    if do_ollama:
        # Ollama might be slow, so we use tqdm loop
        for c in tqdm(chunks, desc="Embedding with Ollama"):
            embeddings_list.append(embed(c, do_ollama=True))
    else:
        # SentenceTransformer handles batching well, but let's show progress
        # embed() handles the batch if passed a list
        embeddings_list = list(embed(chunks, do_ollama=False))

    # Get collection
    collection = vector_db.client.get_or_create_collection(name=vector_db.db_name)

    # Add to collection
    # Chroma handles batching, but we can pass all at once if memory allows
    print("Adding to vector store...")
    collection.add(
        ids=keys,
        documents=chunks,
        embeddings=embeddings_list,
        metadatas=metadatas,
    )
    
    print(f"Successfully ingested {len(chunks)} documents.")
    return 0


def drop_vector_db(vector_db: VectorDb) -> int:
    """
    Drops the collection from the vector db.

    Args:
        vector_db: The dataclass with all the info needed to get the collection.

    Returns:
        0 for success.
    """
    try:
        vector_db.client.delete_collection(name=vector_db.db_name)
        print(f"Collection '{vector_db.db_name}' deleted.")
    except Exception as e:
        print(f"Error deleting collection: {e}")
    return 0


def _build_metadata_filter(entities: List[Entity]) -> Optional[Dict[str, Any]]:
    """
    Constructs a ChromaDB metadata filter from extracted entities.

    Args:
        entities: A list of Entity objects extracted from the query.

    Returns:
        A dictionary representing the metadata filter for ChromaDB.
    """
    filters = []

    for entity in entities:
        if entity.entity_type == "FLIGHT_NUM":
            filters.append({"flight_number": entity.value})
        elif entity.entity_type == "AIRPORT":
            filters.append(
                {"$or": [{"origin": entity.value}, {"destination": entity.value}]}
            )

    if not filters:
        return None

    if len(filters) == 1:
        return filters[0]

    return {"$and": filters}


def query_graph_vector(
    input_query: Union[str, ProcessedQuery], k: int = 5
) -> List[ContextChunk]:
    """
    Retrieves semantically similar records from the vector store.
    
    Args:
        input_query: The query string or ProcessedQuery object.
        k: Number of results to retrieve.

    Returns:
        List of ContextChunk objects.
    """
    client = get_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    query_text = ""
    metadata_filter = None

    if isinstance(input_query, ProcessedQuery):
        query_text = input_query.original_text
        metadata_filter = _build_metadata_filter(input_query.entities)
    else:
        query_text = input_query

    # Embed the query
    query_embedding = embed(query_text, do_ollama=USE_OLLAMA)
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    # Query Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=metadata_filter
    )

    chunks: List[ContextChunk] = []
    
    if not results["ids"]:
        return chunks

    # Unpack results (list of lists)
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

    for i in range(len(ids)):
        # Calculate score (1 - distance approximation)
        score = 1.0 - distances[i] if distances[i] <= 1.0 else 0.0
        
        chunk = ContextChunk(
            id=metas[i].get("id", "unknown"),
            text=docs[i],
            score=round(score, 4),
            source=RetrievalSource.VECTOR,
            metadata=metas[i],
        )
        chunks.append(chunk)

    return chunks


def cli() -> int:
    """
    Quick CLI function to do everything.
    """
    vector_db = VectorDb(client=get_client(), db_name=COLLECTION_NAME)

    parser = argparse.ArgumentParser(description="RAG Vector DB CLI")
    parser.add_argument(
        "-l",
        "--do_load",
        action="store_true",
        help="Flag to load the vector database",
    )
    parser.add_argument(
        "-d",
        "--do_drop",
        action="store_true",
        help="Flag to drop the vector database",
    )
    parser.add_argument(
        "-q",
        "--do_query",
        type=str,
        help="Query string to retrieve from the vector database",
    )
    args = parser.parse_args()

    # Validate arguments
    if not (args.do_drop or args.do_load or args.do_query):
        parser.print_help()
        return 1

    # Execute modes
    if args.do_drop:
        drop_vector_db(vector_db)
    
    if args.do_load:
        load_vector_db(vector_db)

    if args.do_query:
        print(f"Querying: {args.do_query}")
        chunks = query_graph_vector(args.do_query)
        for i, chunk in enumerate(chunks):
            print(f"--- Result {i + 1} (Score: {chunk.score}) ---")
            print(chunk.text)
            print()

    return 0


### ~~~ SCRIPT EXECUTION ~~~ ###
if __name__ == "__main__":
    cli()
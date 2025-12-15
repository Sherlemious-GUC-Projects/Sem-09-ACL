### ~~~ GLOBAL IMPORTS ~~~ ###
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Any, Dict, List, Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dataclasses import asdict
import pandas as pd
import chromadb
import os

### ~~~ LOCAL IMPORTS ~~~ ###
from src.utils.types import ContextChunk, Entity, ProcessedQuery, RetrievalSource
from src.retrieval.util import load_data
from src.utils.constant import CSV_PATH

### ~~~ STATE DEFINITIONS ~~~ ###
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "airline_reviews"
USE_OLLAMA = False


### ~~~ FUNCTION DEFINITIONS ~~~ ###
def get_embedding_model(use_ollama: bool = USE_OLLAMA) -> Embeddings:
    """
    Returns the configured embedding model.
    Args:
        use_ollama: Boolean flag to determine whether to use Ollama embeddings.
                    Defaults to the global USE_OLLAMA constant.
    Returns:
        An instance of an Embeddings model (OllamaEmbeddings or HuggingFaceEmbeddings).
    """
    if use_ollama:
        return OllamaEmbeddings(model="all-minilm")
    else:
        # Uses sentence-transformers/all-MiniLM-L6-v2 locally
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _format_record(row: pd.Series) -> str:
    """
    Formats a CSV row into the target sentence structure.
    Template:
    "A {loyalty} passenger on flight {flight_num} in {class} class experienced a delay of {delay} minutes and rated the food as {food_score}/5."
    Args:
        row: A pandas Series representing a single row from the dataframe.
    Returns:
        A formatted string description of the passenger's experience.
    """
    # Mapping based on KR.md vs Template
    record = {
        "loyalty": row.get("loyalty_program_level", "Unknown"),
        "flight_num": row.get("flight_number", "Unknown"),
        "class": row.get("passenger_class", "Unknown"),
        "delay": row.get("arrival_delay_minutes", 0),
        "food_score": row.get("food_satisfaction_score", 0),
    }
    return (
        f"A {record['loyalty']} passenger on flight {record['flight_num']} "
        f"in {record['class']} class experienced a delay of {record['delay']} "
        f"minutes and rated the food as {record['food_score']}/5."
    )


def _load_and_process_data(csv_path: str) -> List[Document]:
    """
    Reads the CSV and converts rows to LangChain Documents.

    Args:
        csv_path: The file path to the CSV data.

    Returns:
        A list of LangChain Document objects containing processed text and metadata.
    """
    try:
        df = load_data(csv_path)
    except FileNotFoundError:
        print(f"WARNING: CSV not found at {csv_path}. Skipping ingestion.")
        return []

    documents: List[Document] = []

    for _, row in df.iterrows():
        text = _format_record(row)

        # Construct Metadata
        metadata = {
            "id": str(row.get("feedback_ID", row.get("record_locator", "unknown"))),
            "flight_number": str(row.get("flight_number", "")),
            "origin": str(row.get("origin_station_code", "")),
            "destination": str(row.get("destination_station_code", "")),
            "date": str(row.get("flight_date", "")),
            "loyalty": str(row.get("loyalty_program_level", "")),
            "class": str(row.get("passenger_class", "")),
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def get_vector_store(
    persist_directory: str = VECTOR_DB_PATH, collection_name: str = COLLECTION_NAME
) -> Chroma:
    """
    Initializes and returns the Chroma vector store.
    If the collection is empty, it attempts to load data from the CSV.
    Args:
        persist_directory: Directory where the vector store is persisted.
        collection_name: Name of the Chroma collection.
    Returns:
        An initialized Chroma vector store instance.
    """
    embedding_function = get_embedding_model()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    # Simple check: If empty, ingest
    # Note: This checks the number of IDs in the underlying collection
    if not vector_store.get()["ids"]:
        print("Vector store is empty. Ingesting data...")
        docs = _load_and_process_data(CSV_PATH)
        if docs:
            # Add in batches to avoid hitting limits if necessary,
            # though Chroma handles reasonable sizes well.
            vector_store.add_documents(docs)
            print(f"Ingested {len(docs)} documents.")

    return vector_store


def _build_metadata_filter(entities: List[Entity]) -> Optional[Dict[str, Any]]:
    """
    Constructs a ChromaDB metadata filter from extracted entities.
    Args:
        entities: A list of Entity objects extracted from the query.
    Returns:
        A dictionary representing the metadata filter for ChromaDB, or None if no relevant entities are found.
    """
    filters = []

    for entity in entities:
        if entity.entity_type == "FLIGHT_NUM":
            filters.append({"flight_number": entity.value})
        elif entity.entity_type == "AIRPORT":
            # Airports could be origin or destination.
            # Chroma $or syntax for same field is easy, but across fields requires $or at top level
            # For simplicity, let's search both or strict match if we knew context.
            # Here we'll try to match either origin OR destination
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
    Supports structured filtering if ProcessedQuery is provided.
    Args:
        input_query: The query string or a ProcessedQuery object containing the query and entities.
        k: The number of results to retrieve.

    Returns:
        A list of ContextChunk objects representing the retrieved documents.
    """
    store = get_vector_store()

    query_text = ""
    metadata_filter = None

    if isinstance(input_query, ProcessedQuery):
        query_text = input_query.original_text
        metadata_filter = _build_metadata_filter(input_query.entities)
    else:
        query_text = input_query

    # Perform Similarity Search
    # Note: If metadata_filter is complex, ensure Chroma version supports it.
    results = store.similarity_search_with_score(
        query_text, k=k, filter=metadata_filter
    )

    chunks = []
    for doc, score in results:
        # Chroma distance score: lower is better (0=exact).
        # Convert to similarity (1 / (1 + distance)) or just 1 - distance if normalized.
        # Often Chroma returns L2 distance.
        # For interface consistency (0.0-1.0), let's approximate:
        # similarity = 1 - score (if score is cosine distance)
        # We'll stick to raw score or a simple inversion for relevance.
        relevance = 1.0 - score if score <= 1.0 else 0.0

        chunk = ContextChunk(
            id=doc.metadata.get("id", "unknown"),
            text=doc.page_content,
            score=round(relevance, 4),
            source=RetrievalSource.VECTOR,
            metadata=doc.metadata,
        )
        chunks.append(chunk)

    return chunks


def main() -> None:
    """
    Main function for standalone execution and testing.
    """
    print("Initializing Vector DB...")
    vs = get_vector_store()
    print(f"Vector DB contains {len(vs.get()['ids'])} documents.")


### ~~~ SCRIPT EXECUTION ~~~ ###
if __name__ == "__main__":
    main()

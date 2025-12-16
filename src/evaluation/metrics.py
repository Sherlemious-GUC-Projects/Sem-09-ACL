### ~~~ GLOBAL IMPORTS ~~~ ###
import numpy as np
from numpy.linalg import norm

### ~~~ LOCAL IMPORTS ~~~ ###
from src.retrieval.vector import embed  # Reusing the project's embedding function
from src.utils.constant import USE_OLLAMA

### ~~~ FUNCTION DEFINITIONS ~~~ ###


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculates the cosine semantic similarity between two text strings.
    Uses the project's configured embedding model (SentenceTransformer or Ollama).

    Args:
        text1: The first text string.
        text2: The second text string.

    Returns:
        A float representing the cosine similarity, ranging from -1.0 to 1.0.
        Returns 0.0 if either text cannot be embedded or results in a zero vector.
    """
    if not text1 or not text2:
        return 0.0

    try:
        # Embed texts using the project's configured embedding function
        embedding1 = embed(text1, do_ollama=USE_OLLAMA)
        embedding2 = embed(text2, do_ollama=USE_OLLAMA)

        # Ensure embeddings are numpy arrays
        if isinstance(
            embedding1, list
        ):  # Handles the case where embed returns list of arrays for single string due to Ollama batching quirk
            embedding1 = embedding1[0] if embedding1 else np.array([])
        if isinstance(embedding2, list):
            embedding2 = embedding2[0] if embedding2 else np.array([])

        if (
            embedding1.ndim == 0
            or embedding2.ndim == 0
            or not embedding1.size
            or not embedding2.size
        ):
            return 0.0  # Handle empty embeddings

        # Calculate cosine similarity
        # If the vectors are identical or very similar, their dot product will be high.
        # If they are orthogonal (no similarity), dot product is zero.
        # If they are opposite, dot product is negative.

        dot_product = np.dot(embedding1, embedding2)
        norm_a = norm(embedding1)
        norm_b = norm(embedding2)

        if norm_a == 0 or norm_b == 0:
            return 0.0  # Avoid division by zero for zero vectors

        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0

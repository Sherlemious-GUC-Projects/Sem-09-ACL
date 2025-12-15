from typing import List, Optional
from utils.types import ContextChunk, RetrievalSource


def deduplicate_contexts(contexts: List[ContextChunk]) -> List[ContextChunk]:
    """
    Remove duplicate contexts based on ID, keeping highest-scored version.

    Args:
        contexts: List of ContextChunk objects (may contain duplicates)

    Returns:
        Deduplicated list of ContextChunk objects
    """
    if not contexts:
        return []

    # Sort by score (descending) to keep best duplicates first
    sorted_contexts = sorted(contexts, key=lambda x: x.score, reverse=True)

    seen_ids = set()
    deduplicated = []

    for ctx in sorted_contexts:
        if ctx.id not in seen_ids:
            deduplicated.append(ctx)
            seen_ids.add(ctx.id)

    return deduplicated


def rank_contexts(contexts: List[ContextChunk]) -> List[ContextChunk]:
    """
    Rank contexts by relevance score (descending).

    Args:
        contexts: List of ContextChunk objects

    Returns:
        Sorted list with highest-scored contexts first
    """
    return sorted(contexts, key=lambda x: x.score, reverse=True)


def balance_sources(
    contexts: List[ContextChunk], max_cypher: int = 5, max_vector: int = 5
) -> List[ContextChunk]:
    """
    Ensure balanced representation from both retrieval sources.

    Args:
        contexts: List of ContextChunk objects
        max_cypher: Maximum number of Cypher results to include
        max_vector: Maximum number of Vector results to include

    Returns:
        Balanced list of ContextChunk objects
    """
    cypher_results = [c for c in contexts if c.source == RetrievalSource.CYPHER]
    vector_results = [c for c in contexts if c.source == RetrievalSource.VECTOR]

    # Take top N from each source
    balanced = cypher_results[:max_cypher] + vector_results[:max_vector]

    # Re-sort by score
    return sorted(balanced, key=lambda x: x.score, reverse=True)


def truncate_to_token_limit(
    contexts: List[ContextChunk],
    max_tokens: int = 4000,
    tokens_per_char: float = 0.25,  # Rough estimate: 4 chars per token
) -> List[ContextChunk]:
    """
    Truncate context list to fit within token limit.

    Args:
        contexts: List of ContextChunk objects (should be pre-sorted by score)
        max_tokens: Maximum tokens allowed
        tokens_per_char: Estimated tokens per character

    Returns:
        Truncated list that fits within token limit
    """
    total_chars = 0
    max_chars = int(max_tokens / tokens_per_char)
    truncated = []

    for ctx in contexts:
        ctx_chars = len(ctx.text)
        if total_chars + ctx_chars <= max_chars:
            truncated.append(ctx)
            total_chars += ctx_chars
        else:
            break

    return truncated


def process_contexts(
    contexts: List[ContextChunk],
    max_tokens: Optional[int] = None,
    balance: bool = False,
) -> List[ContextChunk]:
    """
    Complete context processing pipeline.

    Args:
        contexts: Raw list of ContextChunk objects
        max_tokens: Optional token limit for truncation
        balance: Whether to balance Cypher/Vector sources

    Returns:
        Processed list of ContextChunk objects
    """
    # Step 1: Deduplicate
    processed = deduplicate_contexts(contexts)

    # Step 2: Balance sources (optional)
    if balance:
        processed = balance_sources(processed)

    # Step 3: Rank by score
    processed = rank_contexts(processed)

    # Step 4: Truncate to fit token limit (optional)
    if max_tokens:
        processed = truncate_to_token_limit(processed, max_tokens)

    return processed


def merge_contexts(
    cypher_contexts: List[ContextChunk], vector_contexts: List[ContextChunk]
) -> List[ContextChunk]:
    """
    Merge contexts from both retrieval methods.

    Args:
        cypher_contexts: Results from Cypher retrieval
        vector_contexts: Results from Vector retrieval

    Returns:
        Merged and processed list
    """
    merged = cypher_contexts + vector_contexts
    return process_contexts(merged)

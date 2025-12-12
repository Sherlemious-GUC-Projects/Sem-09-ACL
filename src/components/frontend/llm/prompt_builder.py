"""Prompt construction utilities for LLM queries."""

from typing import List
from src.utils.types import ContextChunk


SYSTEM_PROMPT = """You are an airline flight information assistant for internal analytics at an airline company. You help analyze flight operations, delays, passenger satisfaction, and journey patterns. Your role is to provide accurate, data-driven insights based on the company's knowledge graph."""

INSTRUCTIONS = """INSTRUCTIONS:
- Answer the user's question using ONLY the provided context information
- If the information needed to answer the question is not in the context, explicitly state "I don't have that information in the available data"
- Be concise and factual in your responses
- When possible, cite which context items you used by referencing their numbers (e.g., "According to item #1...")
- Do not make assumptions or extrapolate beyond the provided data
- Focus on actionable insights relevant to airline operations"""


def build_context_section(contexts: List[ContextChunk]) -> str:
    """
    Build the context section of the prompt from ContextChunk objects.

    Args:
        contexts: List of ContextChunk objects (should be processed/deduplicated)

    Returns:
        Formatted context string
    """
    if not contexts:
        return "CONTEXT:\nNo relevant information found in the knowledge graph."

    context_lines = ["CONTEXT:"]
    for i, ctx in enumerate(contexts, 1):
        source_label = f"[{ctx.source}]"
        score_label = f"(relevance: {ctx.score:.2f})"
        context_lines.append(
            f"{i}. {source_label} {score_label} {ctx.text}"
        )

    return "\n".join(context_lines)


def build_prompt(user_query: str, contexts: List[ContextChunk]) -> str:
    """
    Build complete structured prompt for LLM.

    Args:
        user_query: The user's question
        contexts: List of processed ContextChunk objects

    Returns:
        Complete prompt string
    """
    context_section = build_context_section(contexts)

    prompt = f"""{SYSTEM_PROMPT}

{context_section}

{INSTRUCTIONS}

USER QUERY: {user_query}

ANSWER:"""

    return prompt


def build_system_and_user_messages(
    user_query: str,
    contexts: List[ContextChunk]
) -> tuple[str, str]:
    """
    Build separate system and user messages (for chat-based APIs like Claude).

    Args:
        user_query: The user's question
        contexts: List of processed ContextChunk objects

    Returns:
        Tuple of (system_message, user_message)
    """
    context_section = build_context_section(contexts)

    user_message = f"""{context_section}

{INSTRUCTIONS}

USER QUERY: {user_query}"""

    return SYSTEM_PROMPT, user_message


def truncate_prompt_to_limit(
    prompt: str,
    max_chars: int,
    keep_instructions: bool = True
) -> str:
    """
    Truncate prompt if it exceeds character limit.

    Args:
        prompt: The full prompt
        max_chars: Maximum characters allowed
        keep_instructions: Whether to preserve instructions section

    Returns:
        Truncated prompt with warning if truncation occurred
    """
    if len(prompt) <= max_chars:
        return prompt

    if keep_instructions:
        # Try to preserve system prompt and instructions
        parts = prompt.split("CONTEXT:")
        if len(parts) == 2:
            prefix = parts[0] + "CONTEXT:"
            remaining_chars = max_chars - len(prefix) - 200  # Buffer for end

            # Truncate context section
            context_section = parts[1].split("INSTRUCTIONS:")[0]
            truncated_context = context_section[:remaining_chars]

            # Find last complete line
            last_newline = truncated_context.rfind("\n")
            if last_newline > 0:
                truncated_context = truncated_context[:last_newline]

            # Reconstruct
            rest = "INSTRUCTIONS:" + prompt.split("INSTRUCTIONS:")[1]
            return f"{prefix}{truncated_context}\n[... context truncated ...]\n\n{rest}"

    # Fallback: simple truncation
    return prompt[:max_chars] + "\n\n[... truncated ...]"

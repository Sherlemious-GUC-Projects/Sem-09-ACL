from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelSpec:
    """Specification for an LLM model."""

    name: str
    provider: str  # "gemini", "claude", "groq"
    context_window: int  # Maximum context tokens
    input_price_per_1k: float  # USD per 1000 input tokens
    output_price_per_1k: float  # USD per 1000 output tokens
    display_name: str  # Human-readable name for UI


# Model specifications database
# Pricing as of December 2024 - update periodically
MODEL_SPECS: Dict[str, ModelSpec] = {
    # Google Gemini Models (2025)
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        provider="gemini",
        context_window=1_048_576,
        input_price_per_1k=0.0002,
        output_price_per_1k=0.0008,
        display_name="Gemini 2.5 Flash (Fast & Intelligent)",
    ),
    "gemini-2.5-flash-lite": ModelSpec(
        name="gemini-2.5-flash-lite",
        provider="gemini",
        context_window=1_048_576,
        input_price_per_1k=0.0001,
        output_price_per_1k=0.0004,
        display_name="Gemini 2.5 Flash-Lite (Ultra Fast)",
    ),
    # Groq Models (Free tier available)
    "llama-3.1-8b-instant": ModelSpec(
        name="llama-3.1-8b-instant",
        provider="groq",
        context_window=131_072,
        input_price_per_1k=0.00005,
        output_price_per_1k=0.00008,
        display_name="Llama 3.1 8B (Ultra Fast)",
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    """
    Get model specification by name.

    Args:
        model_name: The model identifier

    Returns:
        ModelSpec object

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(MODEL_SPECS.keys())}"
        )
    return MODEL_SPECS[model_name]


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of an LLM API call.

    Args:
        model_name: The model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    spec = get_model_spec(model_name)
    input_cost = (input_tokens / 1000) * spec.input_price_per_1k
    output_cost = (output_tokens / 1000) * spec.output_price_per_1k
    return input_cost + output_cost


def get_all_model_names() -> list[str]:
    """Get list of all supported model names."""
    return list(MODEL_SPECS.keys())


def get_models_by_provider(provider: str) -> list[str]:
    """
    Get all models for a specific provider.

    Args:
        provider: "gemini", "claude", or "groq"

    Returns:
        List of model names
    """
    return [name for name, spec in MODEL_SPECS.items() if spec.provider == provider]


# Model presets for different use cases
MODEL_PRESETS = {
    "Fast": "gemini-2.5-flash-lite",
    "Balanced": "gemini-2.5-flash",
    "High Quality": "claude-3-5-sonnet-20241022",
    "Most Intelligent": "gemini-3-pro-preview",
    "Advanced Thinking": "gemini-2.5-pro",
    "Ultra Fast (Groq)": "llama-3.1-8b-instant",
}

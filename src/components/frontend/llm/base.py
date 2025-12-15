"""Base classes and interfaces for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""

    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str  # Generated text
    model: str  # Model name used
    tokens_used: int  # Total tokens (input + output)
    input_tokens: int  # Input tokens only
    output_tokens: int  # Output tokens only
    cost_usd: float  # Estimated cost in USD
    response_time_ms: float  # Time taken in milliseconds
    raw_response: Dict[str, Any]  # Provider-specific metadata
    cached: bool = False  # Whether this was served from cache


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the LLM provider.

        Args:
            model_name: The specific model to use
            api_key: API key for authentication
        """
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The complete prompt to send to the LLM
            config: Generation configuration (temperature, max_tokens, etc.)

        Returns:
            LLMResponse object

        Raises:
            LLMProviderError: If the API call fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model metadata (name, provider, context_window, etc.)
        """
        pass

    def validate_api_key(self) -> bool:
        """
        Check if the API key is valid.

        Returns:
            True if valid, False otherwise
        """
        return bool(self.api_key and self.api_key.strip())


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class APIKeyMissingError(LLMProviderError):
    """Raised when API key is missing or invalid."""

    pass


class RateLimitError(LLMProviderError):
    """Raised when API rate limit is exceeded."""

    pass


class TokenLimitError(LLMProviderError):
    """Raised when token limit is exceeded."""

    pass


class ModelNotFoundError(LLMProviderError):
    """Raised when the specified model doesn't exist."""

    pass

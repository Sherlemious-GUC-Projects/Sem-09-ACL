"""Groq LLM provider implementation."""

import time
from typing import Dict, Any

from groq import Groq, APIError, RateLimitError as GroqRateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.components.frontend.llm.base import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    APIKeyMissingError,
    RateLimitError,
    LLMProviderError,
)
from src.components.frontend.config.models import get_model_spec, calculate_cost


class GroqProvider(LLMProvider):
    """Groq LLM provider (Llama, Mixtral, Gemma models)."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize Groq provider.

        Args:
            model_name: Model identifier (e.g., "llama-3.1-8b-instant")
            api_key: Groq API key

        Raises:
            APIKeyMissingError: If API key is missing or invalid
        """
        super().__init__(model_name, api_key)

        if not self.validate_api_key():
            raise APIKeyMissingError("Groq API key is missing or invalid")

        # Initialize Groq client
        try:
            self.client = Groq(api_key=self.api_key)
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Groq client: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate response using Groq.

        Args:
            prompt: The complete prompt
            config: Generation configuration

        Returns:
            LLMResponse object

        Raises:
            RateLimitError: If rate limit is exceeded
            LLMProviderError: For other API errors
        """
        start_time = time.time()

        try:
            # Groq uses OpenAI-compatible chat completions API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
            )

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Extract text
            text = response.choices[0].message.content if response.choices else ""

            # Extract token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = (
                response.usage.total_tokens
                if response.usage
                else (input_tokens + output_tokens)
            )

            # Calculate cost
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)

            return LLMResponse(
                text=text,
                model=self.model_name,
                tokens_used=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                response_time_ms=response_time_ms,
                raw_response={
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "model": response.model,
                    "choices": [
                        {
                            "index": c.index,
                            "message": {
                                "role": c.message.role,
                                "content": c.message.content,
                            },
                            "finish_reason": c.finish_reason,
                        }
                        for c in response.choices
                    ],
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    },
                },
            )

        except GroqRateLimitError as e:
            raise RateLimitError(f"Groq rate limit exceeded: {str(e)}")

        except APIError as e:
            error_msg = str(e).lower()

            # Check for specific error types
            if "rate limit" in error_msg or "429" in error_msg:
                raise RateLimitError(f"Groq rate limit exceeded: {str(e)}")

            raise LLMProviderError(f"Groq API error: {str(e)}")

        except Exception as e:
            raise LLMProviderError(f"Unexpected error with Groq: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        spec = get_model_spec(self.model_name)
        return {
            "name": spec.name,
            "provider": "groq",
            "display_name": spec.display_name,
            "context_window": spec.context_window,
            "input_price_per_1k": spec.input_price_per_1k,
            "output_price_per_1k": spec.output_price_per_1k,
        }

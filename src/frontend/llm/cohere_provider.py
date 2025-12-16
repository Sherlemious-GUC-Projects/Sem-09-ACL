import time
from typing import Dict, Any

import cohere
from tenacity import retry, stop_after_attempt, wait_exponential

from frontend.llm.base import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    APIKeyMissingError,
    RateLimitError,
    LLMProviderError,
)
from frontend.config.models import get_model_spec, calculate_cost


class CohereProvider(LLMProvider):
    """Cohere LLM provider."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize Cohere provider.

        Args:
            model_name: Model identifier (e.g., "command-a-reasoning-08-2025")
            api_key: Cohere API key

        Raises:
            APIKeyMissingError: If API key is missing or invalid
        """
        super().__init__(model_name, api_key)

        if not self.validate_api_key():
            raise APIKeyMissingError("Cohere API key is missing or invalid")

        # Initialize Cohere client
        try:
            self.client = cohere.ClientV2(api_key=self.api_key)
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Cohere client: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate response using Cohere.

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
            # Create chat request
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                p=config.top_p,
            )

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Extract text from response
            if not response.message or not response.message.content:
                raise LLMProviderError("No response content generated")

            # Get the text content
            text = ""
            for content_item in response.message.content:
                if hasattr(content_item, "text"):
                    text += content_item.text

            # Extract token usage
            try:
                input_tokens = response.usage.tokens.input_tokens
                output_tokens = response.usage.tokens.output_tokens
                total_tokens = input_tokens + output_tokens
            except (AttributeError, KeyError):
                # Fallback: estimate tokens
                input_tokens = len(prompt) // 4
                output_tokens = len(text) // 4
                total_tokens = input_tokens + output_tokens

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
                raw_response={"message": response.message, "usage": response.usage},
            )

        except cohere.TooManyRequestsError as e:
            raise RateLimitError(f"Cohere rate limit exceeded: {str(e)}")
        except cohere.BadRequestError as e:
            raise LLMProviderError(f"Cohere bad request: {str(e)}")
        except cohere.UnauthorizedError as e:
            raise APIKeyMissingError(f"Cohere authentication failed: {str(e)}")
        except Exception as e:
            raise LLMProviderError(f"Cohere API error: {str(e)}")

    def validate_api_key(self) -> bool:
        """
        Validate the Cohere API key.

        Returns:
            True if valid, False otherwise
        """
        if not self.api_key or not isinstance(self.api_key, str):
            return False

        # Basic validation - Cohere keys are typically 40 characters
        if len(self.api_key) < 20:
            return False

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        spec = get_model_spec(self.model_name)
        return {
            "model": self.model_name,
            "provider": "cohere",
            "context_window": spec.context_window,
            "input_price_per_1k": spec.input_price_per_1k,
            "output_price_per_1k": spec.output_price_per_1k,
            "display_name": spec.display_name,
        }

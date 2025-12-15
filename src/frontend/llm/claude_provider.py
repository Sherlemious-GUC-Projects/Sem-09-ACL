import time
from typing import Dict, Any

from anthropic import Anthropic, APIError, RateLimitError as AnthropicRateLimitError
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


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize Claude provider.

        Args:
            model_name: Model identifier (e.g., "claude-3-5-haiku-20241022")
            api_key: Anthropic API key

        Raises:
            APIKeyMissingError: If API key is missing or invalid
        """
        super().__init__(model_name, api_key)

        if not self.validate_api_key():
            raise APIKeyMissingError("Claude API key is missing or invalid")

        # Initialize Anthropic client
        try:
            self.client = Anthropic(api_key=self.api_key)
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Claude client: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate response using Claude.

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
            # Claude uses system/user message format
            # Extract system message if present
            if "You are an airline" in prompt:
                parts = prompt.split("\n\n", 1)
                system_message = parts[0]
                user_message = parts[1] if len(parts) > 1 else prompt
            else:
                system_message = "You are a helpful AI assistant."
                user_message = prompt

            # Call Claude API
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                system=system_message,
                messages=[{"role": "user", "content": user_message}],
            )

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Extract text
            text = response.content[0].text if response.content else ""

            # Extract token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
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
                raw_response={
                    "id": response.id,
                    "type": response.type,
                    "role": response.role,
                    "content": [
                        {"type": c.type, "text": c.text if hasattr(c, "text") else ""}
                        for c in response.content
                    ],
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            )

        except AnthropicRateLimitError as e:
            raise RateLimitError(f"Claude rate limit exceeded: {str(e)}")

        except APIError as e:
            error_msg = str(e).lower()

            # Check for specific error types
            if "rate limit" in error_msg or "429" in error_msg:
                raise RateLimitError(f"Claude rate limit exceeded: {str(e)}")

            raise LLMProviderError(f"Claude API error: {str(e)}")

        except Exception as e:
            raise LLMProviderError(f"Unexpected error with Claude: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        spec = get_model_spec(self.model_name)
        return {
            "name": spec.name,
            "provider": "claude",
            "display_name": spec.display_name,
            "context_window": spec.context_window,
            "input_price_per_1k": spec.input_price_per_1k,
            "output_price_per_1k": spec.output_price_per_1k,
        }

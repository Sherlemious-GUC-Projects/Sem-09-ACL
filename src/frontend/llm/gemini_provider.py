import time
from typing import Dict, Any

import google.generativeai as genai
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


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize Gemini provider.

        Args:
            model_name: Model identifier (e.g., "gemini-1.5-flash")
            api_key: Google API key

        Raises:
            APIKeyMissingError: If API key is missing or invalid
        """
        super().__init__(model_name, api_key)

        if not self.validate_api_key():
            raise APIKeyMissingError("Gemini API key is missing or invalid")

        # Configure the SDK
        genai.configure(api_key=self.api_key)

        # Initialize model
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Gemini model: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate response using Gemini.

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
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                top_p=config.top_p,
            )

            # Generate response
            response = self.model.generate_content(
                prompt, generation_config=generation_config
            )

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Extract text
            if not response.candidates:
                raise LLMProviderError("No response candidates generated")

            text = response.text

            # Extract token usage
            # Gemini returns usage metadata
            try:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
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
                raw_response={
                    "candidates": [
                        {
                            "content": (
                                c.content.parts[0].text if c.content.parts else ""
                            ),
                            "finish_reason": c.finish_reason,
                            "safety_ratings": [
                                {"category": r.category, "probability": r.probability}
                                for r in c.safety_ratings
                            ],
                        }
                        for c in response.candidates
                    ],
                    "usage_metadata": {
                        "prompt_token_count": input_tokens,
                        "candidates_token_count": output_tokens,
                        "total_token_count": total_tokens,
                    },
                },
            )

        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limit errors
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                raise RateLimitError(f"Gemini rate limit exceeded: {str(e)}")

            # Re-raise other errors
            raise LLMProviderError(f"Gemini API error: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        spec = get_model_spec(self.model_name)
        return {
            "name": spec.name,
            "provider": "gemini",
            "display_name": spec.display_name,
            "context_window": spec.context_window,
            "input_price_per_1k": spec.input_price_per_1k,
            "output_price_per_1k": spec.output_price_per_1k,
        }

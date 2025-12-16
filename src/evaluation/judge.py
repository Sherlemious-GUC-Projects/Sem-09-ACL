### ~~~ GLOBAL IMPORTS ~~~ ###
import json
import re
from typing import Dict, Any, List
import logging

### ~~~ LOCAL IMPORTS ~~~ ###
from frontend.llm.groq_provider import GroqProvider
from frontend.llm.base import LLMConfig, APIKeyMissingError

### ~~~ STATE DEFINITIONS ~~~ ###
logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT_TEMPLATE = """
You are a strict judge evaluating the faithfulness of an AI-generated answer given a specific context.

CONTEXT:
{context}

ANSWER:
{answer}

INSTRUCTIONS:
1. Break down the ANSWER into individual atomic statements.
2. For each statement, determine if it is directly supported by the CONTEXT.
   - "Supported": The statement is explicitly confirmed by the context.
   - "Contradicted": The context explicitly contradicts the statement.
   - "Not Mentioned": The statement is not present in the context.
3. Calculate a faithfulness score: (Number of Supported Statements) / (Total Number of Statements).
   - If there are no statements, the score is 1.0 (trivial truth).

OUTPUT FORMAT:
Return a valid JSON object with the following structure:
{
    "statements": [
        {
            "text": "Statement 1 text",
            "verdict": "Supported" | "Contradicted" | "Not Mentioned",
            "reason": "Reasoning here"
        }
    ],
    "faithfulness_score": 0.8,
    "reasoning": "Overall explanation"
}

RETURN ONLY THE JSON STRING. NO MARKDOWN.
"""

ANSWER_RELEVANCE_PROMPT_TEMPLATE = """
You are a strict judge evaluating the relevance of an AI-generated answer to a user question.

QUESTION:
{question}

ANSWER:
{answer}

INSTRUCTIONS:
Rate the relevance of the answer on a scale of 1 to 5.
1: Completely irrelevant.
5: Perfectly addresses the question.

OUTPUT FORMAT:
Return a valid JSON object:
{
    "score": 4,
    "reasoning": "Explanation of the score"
}

RETURN ONLY THE JSON STRING. NO MARKDOWN.
"""

CONTEXT_RELEVANCE_PROMPT_TEMPLATE = """
You are a strict judge evaluating the relevance of retrieved context chunks to a user question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
Rate the relevance of the retrieved context on a scale of 1 to 5.
1: Context contains no relevant information.
5: Context contains all necessary information to answer the question perfectly.

OUTPUT FORMAT:
Return a valid JSON object:
{
    "score": 3,
    "reasoning": "Explanation of the score"
}

RETURN ONLY THE JSON STRING. NO MARKDOWN.
"""

### ~~~ FUNCTION DEFINITIONS ~~~ ###


class JudgeProvider:
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize the JudgeProvider with a Groq model.
        Using a large model (70b) is recommended for better reasoning.
        """
        self.model_name = model_name
        self.api_key = api_key
        try:
            self.provider = GroqProvider(model_name=model_name, api_key=api_key)
        except APIKeyMissingError:
            logger.error("Groq API key missing for JudgeProvider.")
            raise

        # Configuration for deterministic output
        self.config = LLMConfig(
            temperature=0.0,
            max_tokens=4096,  # Sufficient for detailed reasoning
            top_p=1.0,
        )

    def _clean_json_response(self, text: str) -> Dict[str, Any]:
        """
        Clean and parse JSON from the LLM response.
        Handles markdown code blocks (```json ... ```).
        """
        # Remove markdown code blocks
        clean_text = re.sub(r"```json\s*", "", text)
        clean_text = re.sub(r"```\s*", "", clean_text)
        clean_text = clean_text.strip()

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Judge: {text}")
            # Fallback: try to find the first '{' and last '}'
            try:
                start = clean_text.find("{")
                end = clean_text.rfind("}") + 1
                if start != -1 and end != -1:
                    return json.loads(clean_text[start:end])
            except Exception:
                pass
            raise ValueError(f"Invalid JSON response from Judge: {e}")

    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """
        Evaluates how faithful the answer is to the provided context.
        Returns a score between 0.0 and 1.0.
        """
        prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(context=context, answer=answer)
        response = self.provider.generate(prompt, self.config)

        try:
            data = self._clean_json_response(response.text)
            return float(data.get("faithfulness_score", 0.0))
        except Exception as e:
            logger.error(f"Error in evaluate_faithfulness: {e}")
            return 0.0

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Evaluates how relevant the answer is to the question.
        Returns a score between 1.0 and 5.0.
        """
        prompt = ANSWER_RELEVANCE_PROMPT_TEMPLATE.format(
            question=question, answer=answer
        )
        response = self.provider.generate(prompt, self.config)

        try:
            data = self._clean_json_response(response.text)
            return float(data.get("score", 1.0))
        except Exception as e:
            logger.error(f"Error in evaluate_answer_relevance: {e}")
            return 1.0

    def evaluate_context_relevance(self, question: str, context: List[str]) -> float:
        """
        Evaluates the quality of the retrieved context chunks.
        Returns a score between 1.0 and 5.0.
        """
        # Join list of strings into a single context block
        context_text = "\n\n".join(context)

        prompt = CONTEXT_RELEVANCE_PROMPT_TEMPLATE.format(
            question=question, context=context_text
        )
        response = self.provider.generate(prompt, self.config)

        try:
            data = self._clean_json_response(response.text)
            return float(data.get("score", 1.0))
        except Exception as e:
            logger.error(f"Error in evaluate_context_relevance: {e}")
            return 1.0

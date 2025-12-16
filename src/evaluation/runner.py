### ~~~ GLOBAL IMPORTS ~~~ ###
import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

### ~~~ LOCAL IMPORTS ~~~ ###
from evaluation.types import (
    TestCase,
    EvaluationResult,
    LLMResponseMetrics,
    JudgeMetrics,
    AccuracyMetrics,
    RetrievalStrategy,
)
from evaluation.judge import JudgeProvider
from evaluation.metrics import calculate_semantic_similarity

# LLM Providers from frontend
from frontend.llm.base import LLMConfig, APIKeyMissingError, LLMProviderError
from frontend.llm.gemini_provider import GeminiProvider
from frontend.llm.groq_provider import GroqProvider

# Retrieval modules
from retrieval.vector import query_graph_vector
from retrieval.graph_retrieval import query_graph_cypher

# Ingestion for ProcessedQuery
from ingestion.main import (
    main as process_user_query_for_kg,
)  # Alias to avoid confusion

# Types for retrieval
from utils.types import ContextChunk, ProcessedQuery

### ~~~ STATE DEFINITIONS ~~~ ###
load_dotenv()  # Load environment variables

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# logging.disable(logging.CRITICAL)  # so we can have tqdm!

# --- Configuration ---
GOLD_STANDARD_PATH = "src/evaluation/data/gold_standard.json"
RESULTS_CSV_PATH = (
    "results/evaluation_results.csv"  # Will create this directory if it doesn't exist
)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Target Models for evaluation
# Ensure these match the model_name expected by their respective providers
TARGET_LLMS = {
    "groq_llama_3_1_8b": (
        "llama-3.1-8b-instant",
        GroqProvider,
        GROQ_API_KEY,
    ),
    "gemini_2_5_flash": (
        "gemini-2.5-flash",
        GeminiProvider,
        GEMINI_API_KEY,
    ),
    "gemini_2_5_flash_lite": (
        "gemini-2.5-flash-lite",
        GeminiProvider,
        GEMINI_API_KEY,
    ),
}

# Judge Model (using a more capable Groq model for judging)
JUDGE_MODEL_NAME = "llama-3.1-8b-instant"
JUDGE_API_KEY = GROQ_API_KEY

# LLM Config for generating answers (not for judging)
ANSWER_GEN_LLM_CONFIG = LLMConfig(
    temperature=0.1,  # Keep it low for consistent answers
    max_tokens=1024,
    top_p=1.0,
)

### ~~~ FUNCTION DEFINITIONS ~~~ ###


def load_test_cases(file_path: str) -> List[TestCase]:
    """Loads test cases from a JSON file."""
    if not os.path.exists(file_path):
        logger.error(f"Test cases file not found: {file_path}")
        return []
    with open(file_path, "r") as f:
        data = json.load(f)
    return [TestCase(**item) for item in data]


def initialize_llm_providers(target_llms: Dict[str, Any]) -> Dict[str, Any]:
    """Initializes LLM providers based on configuration."""
    providers = {}
    for llm_alias, (model_name, provider_class, api_key) in target_llms.items():
        if not api_key:
            logger.warning(
                f"API Key for {llm_alias} ({model_name}) is missing. Skipping this model."
            )
            continue
        try:
            providers[llm_alias] = provider_class(
                model_name=model_name, api_key=api_key
            )
            logger.info(f"Initialized {llm_alias} ({model_name}) provider.")
        except APIKeyMissingError:
            logger.error(
                f"API Key for {llm_alias} ({model_name}) is invalid or missing."
            )
        except LLMProviderError as e:
            logger.error(
                f"Failed to initialize {llm_alias} ({model_name}) provider: {e}"
            )
    return providers


def run_evaluation() -> None:
    """Orchestrates the entire evaluation process."""
    test_cases = load_test_cases(GOLD_STANDARD_PATH)
    if not test_cases:
        logger.error("No test cases loaded. Exiting evaluation.")
        return

    # Initialize Judge Provider
    judge_provider: Optional[JudgeProvider] = None
    if JUDGE_API_KEY:
        try:
            judge_provider = JudgeProvider(
                api_key=JUDGE_API_KEY, model_name=JUDGE_MODEL_NAME
            )
            logger.info(f"Initialized JudgeProvider with {JUDGE_MODEL_NAME}.")
        except APIKeyMissingError:
            logger.error(
                "JudgeProvider API Key missing. Qualitative metrics will be skipped."
            )
        except Exception as e:
            logger.error(f"Failed to initialize JudgeProvider: {e}")
            judge_provider = None
    else:
        logger.warning("JUDGE_API_KEY is not set. Qualitative metrics will be skipped.")

    # Initialize Target LLM Providers
    target_llm_providers = initialize_llm_providers(TARGET_LLMS)
    if not target_llm_providers:
        logger.error("No target LLM providers initialized. Exiting evaluation.")
        return

    all_results: List[EvaluationResult] = []

    logger.info(
        f"Starting evaluation for {len(test_cases)} test cases, "
        f"{len(target_llm_providers)} LLMs, and 2 retrieval strategies."
    )

    for i, test_case in tqdm(list(enumerate(test_cases))):
        logger.info(
            f"\n--- Running Test Case {i + 1}/{len(test_cases)}: {test_case.question} ---"
        )

        processed_query: Optional[ProcessedQuery] = None
        try:
            # This is the NLP pre-processing step for KG retrieval
            processed_query = process_user_query_for_kg(test_case.question)
            logger.debug(
                f"Processed query for KG: Intent={processed_query.intent}, Entities={[e.value for e in processed_query.entities]}"
            )
        except Exception as e:
            logger.error(
                f"Failed to process query for KG for '{test_case.question}': {e}. KG retrieval will be skipped for this case."
            )
            # We can't proceed with KG for this test case if processing fails

        for llm_alias, llm_provider in target_llm_providers.items():
            # time.sleep(60)  # To avoid rate limits between test cases
            for retrieval_strategy in [RetrievalStrategy.DENSE, RetrievalStrategy.KG]:
                logger.info(
                    f"  Evaluating {llm_alias} with {retrieval_strategy.value} retrieval..."
                )

                retrieved_context_chunks: List[ContextChunk] = []
                retrieved_context_text = ""
                generated_answer = ""
                llm_metrics_obj: Optional[LLMResponseMetrics] = None

                if retrieval_strategy == RetrievalStrategy.DENSE:
                    try:
                        retrieved_context_chunks = query_graph_vector(
                            test_case.question
                        )
                    except Exception as e:
                        logger.error(
                            f"  Dense retrieval failed for '{test_case.question}': {e}"
                        )
                elif retrieval_strategy == RetrievalStrategy.KG:
                    if processed_query:
                        try:
                            retrieved_context_chunks = query_graph_cypher(
                                processed_query
                            )
                        except Exception as e:
                            logger.error(
                                f"  KG retrieval failed for '{test_case.question}': {e}"
                            )
                    else:
                        logger.warning(
                            f"  Skipping KG retrieval for '{test_case.question}' due to failed query processing."
                        )
                        continue  # Skip to next iteration if KG processing failed

                if not retrieved_context_chunks:
                    logger.warning(
                        f"  No context retrieved for {llm_alias} with {retrieval_strategy.value}."
                    )

                retrieved_context_text = "\n\n".join(
                    [c.text for c in retrieved_context_chunks]
                )

                # Generate Answer
                try:
                    full_prompt = f"Context: {retrieved_context_text}\n\nQuestion: {test_case.question}\n\nAnswer:"
                    llm_response = llm_provider.generate(
                        full_prompt, ANSWER_GEN_LLM_CONFIG
                    )
                    generated_answer = llm_response.text
                    llm_metrics_obj = LLMResponseMetrics(
                        response_time_ms=llm_response.response_time_ms,
                        cost_usd=llm_response.cost_usd,
                        tokens_used=llm_response.tokens_used,
                        generated_text=generated_answer,
                        retrieved_context=[c.text for c in retrieved_context_chunks],
                    )
                except LLMProviderError as e:
                    logger.error(
                        f"  LLM generation failed for {llm_alias} on '{test_case.question}': {e}"
                    )
                    continue  # Skip to next iteration if LLM generation fails
                except Exception as e:
                    logger.error(
                        f"  Unexpected error during LLM generation for {llm_alias} on '{test_case.question}': {e}"
                    )
                    continue

                # --- Collect Judge Metrics ---
                judge_metrics_obj = JudgeMetrics(
                    answer_relevance_score=0.0,  # Default if judge fails/skipped
                    faithfulness_score=0.0,
                    context_relevance_score=0.0,
                )
                if judge_provider:
                    try:
                        judge_metrics_obj.answer_relevance_score = (
                            judge_provider.evaluate_answer_relevance(
                                test_case.question, generated_answer
                            )
                        )
                        judge_metrics_obj.faithfulness_score = (
                            judge_provider.evaluate_faithfulness(
                                retrieved_context_text, generated_answer
                            )
                        )
                        judge_metrics_obj.context_relevance_score = (
                            judge_provider.evaluate_context_relevance(
                                test_case.question,
                                [c.text for c in retrieved_context_chunks],
                            )
                        )  # Pass original question to judge for context relevance
                    except Exception as e:
                        logger.error(
                            f"  Judge evaluation failed for {llm_alias} on '{test_case.question}': {e}"
                        )
                else:
                    logger.warning(
                        "  JudgeProvider not initialized. Skipping qualitative metrics."
                    )

                # --- Collect Accuracy Metrics (Semantic Similarity) ---
                semantic_similarity_score = calculate_semantic_similarity(
                    test_case.ground_truth_answer, generated_answer
                )
                accuracy_metrics_obj = AccuracyMetrics(
                    semantic_similarity=semantic_similarity_score
                )

                # --- Store Results ---
                all_results.append(
                    EvaluationResult(
                        test_case=test_case,
                        llm_name=llm_alias,
                        retrieval_strategy=retrieval_strategy,
                        llm_metrics=llm_metrics_obj,
                        judge_metrics=judge_metrics_obj,
                        accuracy_metrics=accuracy_metrics_obj,
                        raw_judge_output=None,  # Not storing raw judge output for now to keep results leaner
                    )
                )
                logger.info(
                    f"  Completed evaluation for {llm_alias} with {retrieval_strategy.value}."
                )

    # Export Results
    export_results(all_results, RESULTS_CSV_PATH)
    logger.info(f"Evaluation complete. Results saved to {RESULTS_CSV_PATH}")


def export_results(results: List[EvaluationResult], output_path: str):
    """Exports evaluation results to a CSV file."""
    if not results:
        logger.warning("No results to export.")
        return

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "test_case_question",
        "test_case_ground_truth_answer",
        "llm_name",
        "retrieval_strategy",
        "generated_answer",
        "retrieved_context",
        "llm_response_time_ms",
        "llm_cost_usd",
        "llm_tokens_used",
        "judge_answer_relevance_score",
        "judge_faithfulness_score",
        "judge_context_relevance_score",
        "accuracy_semantic_similarity",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "test_case_question": result.test_case.question,
                    "test_case_ground_truth_answer": result.test_case.ground_truth_answer,
                    "llm_name": result.llm_name,
                    "retrieval_strategy": result.retrieval_strategy.value,
                    "generated_answer": result.llm_metrics.generated_text,
                    "retrieved_context": json.dumps(
                        result.llm_metrics.retrieved_context
                    ),  # Store as JSON string
                    "llm_response_time_ms": result.llm_metrics.response_time_ms,
                    "llm_cost_usd": result.llm_metrics.cost_usd,
                    "llm_tokens_used": result.llm_metrics.tokens_used,
                    "judge_answer_relevance_score": result.judge_metrics.answer_relevance_score,
                    "judge_faithfulness_score": result.judge_metrics.faithfulness_score,
                    "judge_context_relevance_score": result.judge_metrics.context_relevance_score,
                    "accuracy_semantic_similarity": result.accuracy_metrics.semantic_similarity,
                }
            )
    logger.info(f"Results exported to {output_path}")


### ~~~ SCRIPT EXECUTION ~~~ ###
if __name__ == "__main__":
    run_evaluation()

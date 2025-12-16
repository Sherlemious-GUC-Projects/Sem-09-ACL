### ~~~ CUSTOM TYPES ~~~ ###
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, TypeAlias
from enum import Enum


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    KG = "kg"


@dataclass
class TestCase:
    question: str
    ground_truth_answer: str
    ground_truth_context: str


@dataclass
class LLMResponseMetrics:
    response_time_ms: float
    cost_usd: float
    tokens_used: int
    generated_text: str
    retrieved_context: List[str]  # Context that was actually passed to the LLM


@dataclass
class JudgeMetrics:
    answer_relevance_score: float  # 1-5
    faithfulness_score: float  # 0-1
    context_relevance_score: float  # 1-5


@dataclass
class AccuracyMetrics:
    semantic_similarity: float  # 0-1


@dataclass
class EvaluationResult:
    test_case: TestCase
    llm_name: str
    retrieval_strategy: RetrievalStrategy
    llm_metrics: LLMResponseMetrics
    judge_metrics: JudgeMetrics
    accuracy_metrics: AccuracyMetrics
    raw_judge_output: Optional[Dict[str, Any]] = (
        None  # Store raw judge output for debugging
    )


# Type alias for the list of test cases
TestCases: TypeAlias = List[TestCase]

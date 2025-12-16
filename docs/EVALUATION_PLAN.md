# Evaluation Strategy & Benchmarking Plan

## 1. Objective
To systematically benchmark the performance of the Retrieval Augmented Generation (RAG) system, specifically comparing:
1.  **LLM Performance:** Comparing different models (e.g., Gemini, Claude, Llama 3) on accuracy, cost, and quality.
2.  **Retrieval Strategies:** Evaluating the effectiveness of **Dense Vector Retrieval** vs. **Knowledge Graph (KG) Retrieval**.

## 2. The Evaluation Pipeline

The evaluation pipeline is designed as a standalone module (`src/evaluation`) that runs a curated test suite against the application logic and uses a "Judge LLM" to score the results.

### 2.1 Core Components

1.  **Test Suite (`gold_standard.json`)**: A curated dataset containing:
    *   `question`: The user query.
    *   `ground_truth_answer`: The ideal human-written answer.
    *   `ground_truth_context`: (Optional) The specific facts/chunks that *should* be retrieved.

2.  **The Runner (`runner.py`)**:
    *   Iterates through the Test Suite.
    *   Configures the Retrieval Mode (KG, Dense, or Hybrid).
    *   Calls the Target LLM (Gemini, Claude, etc.) to generate an answer.
    *   Captures System Metrics (Latency, Tokens, Cost).
    *   Passes inputs/outputs to the **Evaluator**.

3.  **The Evaluator (Judge LLM)**:
    *   A robust, fast model (e.g., `Llama-3-70b` via Groq) tasked with scoring the output based on specific metrics.

---

## 3. Metrics

We classify metrics into three categories: **System**, **Retrieval Quality**, and **Generation Quality**.

### 3.1 System Metrics (Quantitative)
*Derived directly from `src/frontend/llm` infrastructure.*

*   **Latency (ms):** End-to-end time for retrieval + generation.
*   **Cost ($):** Based on token usage and model pricing (`MODEL_SPECS`).
*   **Token Usage:** Input/Output token counts.

### 3.2 Generation Quality (Qualitative via Judge)

#### A. Answer Relevance
*   **Question:** Does the answer directly address the user's prompt?
*   **Method:** Judge LLM returns a score (1-5).

#### B. Semantic Similarity (Accuracy)
*   **Question:** How close is the generated answer to the *Ground Truth*?
*   **Method:** Embedding-based Cosine Similarity (using the project's embedding model).
*   **Goal:** Capture the "vibe" and key meaning without being penalized for phrasing (unlike ROUGE/BLEU).

#### C. Faithfulness (Groundedness)
*   **Question:** Is the answer derived *solely* from the retrieved context (no hallucinations)?
*   **Method:** Multi-step "Atomic Fact Verification":
    1.  **Decomposition:** Break the generated answer into individual atomic statements (e.g., "Einstein was born in Germany").
    2.  **Verification:** Check each statement against the *Retrieved Context*.
        *   *Supported*: Found in context.
        *   *Contradicted*: Context says otherwise.
        *   *Not Mentioned*: Not present in context.
    3.  **Calculation:**
        $$ 	ext{Faithfulness Score} = rac{	ext{Number of Supported Statements}}{	ext{Total Statements}} $$

### 3.3 Retrieval Quality (KG vs. Dense)

To compare Knowledge Graph vs. Dense Retrieval, we measure **Context Relevance**:

*   **Question:** Did the retrieval system (KG or Dense) fetch information relevant to the query?
*   **Method:** Judge LLM analyzes the *Retrieved Chunks/Triples*.
    *   *Input:* Query + Retrieved Context.
    *   *Task:* Rate how useful the retrieved information is for answering the query (Binary: Relevant/Irrelevant or Score 1-5).
*   **Comparison:** We run the same queries twice—once with Dense IR, once with KG—and compare the *Context Relevance* scores and downstream *Faithfulness*.

---

## 4. Implementation Plan

### Phase 1: Infrastructure Setup
- [ ] Create `src/evaluation` directory structure.
- [ ] Define `EvaluationResult` data class (to hold all metrics).
- [ ] Implement `JudgeProvider` (wrapper around Llama3 for scoring).

### Phase 2: Metric Implementation
- [ ] **FaithfulnessService:** Implement the 3-step decomposition and verification logic.
- [ ] **RelevanceService:** Implement context relevance scoring.
- [ ] **SimilarityService:** Implement embedding-based cosine similarity.

### Phase 3: The Runner & Reporting
- [ ] Build `runner.py` to ingest `gold_standard.json` and orchestrate the loop.
- [ ] Implement result aggregation (CSV export).
- [ ] Create a Jupyter Notebook (`notebooks/evaluation_analysis.ipynb`) to visualize the comparison (Bar charts for Cost vs. Accuracy, KG vs. Dense Faithfulness).

---

## 5. Directory Structure
```text
src/
└── evaluation/
    ├── __init__.py
    ├── data/
    │   └── gold_standard.json       # The Test Suite
    ├── judge.py                     # LLM-as-a-Judge Logic
    ├── metrics.py                   # Math & Logic (Similarity, Faithfulness calc)
    └── runner.py                    # Main execution script
```

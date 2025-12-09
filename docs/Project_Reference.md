# Airline Knowledge Graph & Graph-RAG Assistant Project Reference

This document serves as the single source of truth for the project, synthesizing information from the Knowledge Graph specification (`KR.md`), the project description/requirements (`proj_desc.pdf`), and the component breakdown (`components.pdf`).

## 1. Project Overview

The goal of this project is to build an end-to-end **Graph-RAG Travel Assistant** (specifically an "Airline Company Flight Insights Assistant"). The system uses a Neo4j Knowledge Graph (KG) constructed from airline survey data as a grounding mechanism for an LLM-based system.

The project bridges **symbolic reasoning** (Knowledge Graph) and **statistical reasoning** (LLMs) to:

- Reduce hallucination.
- Increase factual accuracy.
- Enhance interpretability.

---

## 2. System Architecture & Components

The system is divided into four main components. Each team member is responsible for one component, being aware of its limitations.

### Component 1: Input Preprocessing

**Goal:** Prepare user input for retrieval.

- **System Overview:** Architecture of the preprocessing pipeline.
- **Intent Classification:** Classify what the user wants to do (e.g., flight search, route analysis, delay queries). This routes the query to the appropriate retrieval strategy.
- **Entity Extraction:** Extract relevant entities (e.g., Airport codes like "ORD", specific Flight numbers, dates) using NER.
- **Input Embedding:** Convert input to vector representation (only needed for Embedding-based retrieval).
- **Error Analysis:** Handling malformed inputs.

### Component 2: Graph Retrieval Layer

**Goal:** Fetch relevant context from the Neo4j Knowledge Graph.
You must implement **two experiments**:

1.  **Baseline (Cypher Only):**
    - Use structured Cypher queries with extracted entities.
    - Requires a library of at least **10 query templates** covering different intents.
2.  **Embeddings (Vector Search):**
    - **Node Embeddings:** Create vector reps for nodes (e.g., using numerical feature vectors for flights/journeys since there is limited text).
    - **Feature Vector Embeddings:** Create vectors for feature combinations. For the Airline theme (mostly numerical), construct text descriptions (e.g., "Journey: X, Class: Y, Delay: Z") or use numerical features directly.
    - Experiment with at least **two different embedding models**.

### Component 3: LLM Layer

**Goal:** Generate the final answer using retrieved context.

- **Context Integration:** Merge results from Baseline (Cypher) and Embeddings retrieval. Remove duplicates/rank results.
- **Structured Prompting:**
  - **Context:** The retrieved KG info.
  - **Persona:** "You are a flight information assistant" (acting from the Airline Company's perspective).
  - **Task:** "Answer using only provided information."
- **Model Comparison:** Compare at least **three models** (e.g., GPT-3.5/4, Gemini, Llama, Mistral).
- **Evaluation:** Qualitative (human eval of naturalness/correctness) and Quantitative (accuracy, tokens, cost).

### Component 4: User Interface (UI)

**Goal:** Demonstrate the system (e.g., using Streamlit).

- **Functionality:**
  - View KG-retrieved context (raw nodes/relationships found).
  - View final LLM answer.
  - (Optional) View executed Cypher queries and graph visualizations.
  - Model Selection Dropdown (real-time comparison).
  - Retrieval Method Selection (Baseline vs. Embeddings).
- **Use Case:** Airline Company Flight Insights Assistant.

---

## 3. Knowledge Graph Specification (The Foundation)

The underlying graph is built from `dbs/Airline_surveys_sample.csv`.

### Schema

#### Nodes

- **`Passenger`**
  - `record_locator` (ID), `loyalty_program_level`, `generation`.
- **`Journey`**
  - `feedback_ID` (ID), `food_satisfaction_score`, `arrival_delay_minutes`, `actual_flown_miles`, `number_of_legs`, `passenger_class`.
- **`Flight`**
  - `flight_number`, `fleet_type_description`.
  - _Constraint:_ Unique by combination of `flight_number` + `fleet_type_description`.
- **`Airport`**
  - `station_code` (ID, e.g., "LHR").

#### Relationships

- `(:Passenger)-[:TOOK]->(:Journey)`
- `(:Journey)-[:ON]->(:Flight)`
- `(:Flight)-[:DEPARTS_FROM]->(:Airport)`
- `(:Flight)-[:ARRIVES_AT]->(:Airport)`

### ETL Pipeline (Existing Implementation)

- **Extract:** Reads CSV data.
- **Transform:** `src/construct.py` cleans data and models entities.
- **Load:** `src/load.py` batches entities into Neo4j using `UNWIND`.
- **Analyze:** `src/query.py` contains pre-defined analytical queries (Top Busiest Routes, Best On-Time Performance, etc.).

---

## 4. Airline Theme Specifics

**Focus:**

- Flight routes, delays, passenger satisfaction, and journey analysis.
- **Perspective:** The assistant acts for the **Airline Company**, gaining insights (e.g., analyzing poorly rated flights), not just a passenger booking bot.

**Constraints:**

- **No textual features:** The dataset is primarily numerical/categorical.
- **Embedding Strategy:** You must construct text from numerical properties (e.g., stringifying metrics) or use numerical feature vectors directly.

**Example Intent/Entities:**

- User: "Flights from ORD to LAX with minimal delays."
- Intent: Flight Search / Delay Analysis.
- Entities: Origin="ORD", Dest="LAX".

---

## 5. Setup & Usage Guide

### Configuration

Create `config.txt` in the project root:

```properties
URI=bolt://localhost:7687
USERNAME=neo4j
PASSWORD=your_secure_password
```

### Running the Pipeline

The `src/main.py` script controls the flow.

1.  **Load Data (First run only):**
    Uncomment `loader()` in `src/main.py` and run:
    ```bash
    python src/main.py
    ```
2.  **Run Analysis:**
    With `loader()` commented out, running `python src/main.py` executes the 6 pre-defined analytical queries and verifies them against `dbs/query_*.json`.

---

## 6. Milestone 3 Deliverables (Deadline: Dec 15th)

**Requirements:**

1.  **System Architecture:** documented design and diagrams.
2.  **Retrieval Strategy:** Documented Cypher templates and Embedding approach.
3.  **LLM Comparison:** Metrics and analysis of the 3+ models.
4.  **Error Analysis:** Where and why the system fails.
5.  **Improvements:** Enhancements made to overcome limitations.

**Submission:**

- GitHub Repository (Private until deadline, then Public/Add Collaborator `csen903w25-sys`).
- Branch named `Milestone3`.
- Link to presentation slides.

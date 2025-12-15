# Airline Insights Assistant - Frontend Module

## Overview

LLM & UI module for the Airline Graph-RAG Assistant. Provides a Streamlit interface with multi-LLM support (Gemini, Claude, Groq) and knowledge graph integration.

## Features

- **8+ LLM Models**: Gemini (Flash, Pro), Claude (Haiku, Sonnet), Groq (Llama, Mixtral, Gemma)
- **Dual Retrieval**: Cypher (structured), Vector (semantic), or Hybrid
- **Interactive UI**: Model selection, parameter tuning, real-time feedback
- **Cost Tracking**: Token usage and cost estimation per query
- **Query History**: Session management with full history

## Quick Setup

### 1. Install Dependencies

```bash
uv sync
uv pip install -e .
```

### 2. Configure API Keys

Copy the example and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_claude_key_here
GROQ_API_KEY=your_groq_key_here
```

### 3. Get Free API Keys

| Provider   | URL                                        | Free Tier              |
| ---------- | ------------------------------------------ | ---------------------- |
| **Groq**   | <https://console.groq.com/>                | Very generous (30 RPM) |
| **Gemini** | <https://makersuite.google.com/app/apikey> | 15 RPM, 1M TPM         |
| **Claude** | <https://console.anthropic.com/>           | $5 credit              |

### 4. Run the App

```powershell
# PowerShell (Windows) - run from project root
$env:PYTHONPATH="{folder path}"; streamlit run src/components/frontend/app.py
```

```bash
# Bash (Linux/Mac) - run from project root
PYTHONPATH=$(pwd) streamlit run src/components/frontend/app.py
```

App opens at `http://localhost:8501`

## Usage

1. **Select Model** in sidebar
2. **Choose Retrieval Method** (Cypher/Vector/Both)
3. **Enter Question** and click Submit
4. **View Results**: Answer, context, metrics, history

### Example Queries

```
What are the most delayed flights from ORD?
Show me passenger satisfaction scores for flights to LHR
Which aircraft types have the best food ratings?
```

## File Structure

```
src/components/frontend/
├── app.py                  # Main entry point + UI
├── llm/                    # LLM providers
│   ├── base.py             # Abstract interfaces
│   ├── gemini_provider.py  # Google Gemini
│   ├── claude_provider.py  # Anthropic Claude
│   ├── groq_provider.py    # Groq (Llama, Mixtral, Gemma)
│   ├── context_processor.py
│   └── prompt_builder.py
├── config/
│   ├── settings.py         # Config loader
│   └── models.py           # Model specs & pricing
└── ui/                     # UI components
```

## Troubleshooting

**API Key Error**: Check sidebar for key status, ensure `.env` file exists

**Module Not Found**: Run `uv pip install -e .` from project root, then use this command to run the app:

```powershell
# PowerShell (Windows)
$env:PYTHONPATH="{folder Path}"; streamlit run src/components/frontend/app.py

# Bash (Linux/Mac)
PYTHONPATH=$(pwd) streamlit run src/components/frontend/app.py
```

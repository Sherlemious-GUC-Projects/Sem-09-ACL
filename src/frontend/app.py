from typing import List
import streamlit as st

from utils.types import ContextChunk
from frontend.llm.base import LLMConfig, LLMResponse, LLMProviderError
from frontend.llm.context_processor import process_contexts
from frontend.llm.prompt_builder import build_prompt
from frontend.llm.gemini_provider import GeminiProvider
from frontend.llm.claude_provider import ClaudeProvider
from frontend.llm.groq_provider import GroqProvider
from frontend.llm.cohere_provider import CohereProvider
from frontend.config.settings import get_config
from frontend.config.models import get_model_spec


def mock_generate_response(user_query: str, context: List[ContextChunk]) -> str:
    """
    STUB: Mock implementation of LLM Generation.
    Use this for testing the pipeline without the real logic.
    """
    context_text = "\n".join([c.text for c in context])
    return (
        f"Based on the context: {context_text}, here is the answer to '{user_query}'."
    )


def get_provider(model_name: str):
    """
    Factory function to get the appropriate LLM provider.

    Args:
        model_name: The model identifier

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If model is unknown or API key is missing
    """
    config = get_config()
    spec = get_model_spec(model_name)

    # Select provider based on model
    if spec.provider == "gemini":
        if not config.llm_providers.gemini_api_key:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY environment variable or add it to config.yaml"
            )
        return GeminiProvider(model_name, config.llm_providers.gemini_api_key)

    elif spec.provider == "claude":
        if not config.llm_providers.anthropic_api_key:
            raise ValueError(
                "Claude API key is not configured. Please set ANTHROPIC_API_KEY environment variable or add it to config.yaml"
            )
        return ClaudeProvider(model_name, config.llm_providers.anthropic_api_key)

    elif spec.provider == "groq":
        if not config.llm_providers.groq_api_key:
            raise ValueError(
                "Groq API key is not configured. Please set GROQ_API_KEY environment variable or add it to config.yaml"
            )
        return GroqProvider(model_name, config.llm_providers.groq_api_key)

    elif spec.provider == "cohere":
        if not config.llm_providers.cohere_api_key:
            raise ValueError(
                "Cohere API key is not configured. Please set COHERE_API_KEY environment variable or add it to config.yaml"
            )
        return CohereProvider(model_name, config.llm_providers.cohere_api_key)

    else:
        raise ValueError(f"Unknown provider: {spec.provider}")


def generate_response(
    user_query: str,
    context: List[ContextChunk],
    model_name: str = "gemini-1.5-flash",
    config: LLMConfig | None = None,
) -> LLMResponse:
    """
    Generate LLM response using provided context.

    Args:
        user_query: The user's question
        context: Retrieved context chunks from KG
        model_name: Which LLM model to use
        config: LLM configuration (temperature, max_tokens, etc.)

    Returns:
        LLMResponse with answer, tokens, cost, timing

    Raises:
        LLMProviderError: If the API call fails
        ValueError: If model is unknown or API key missing
    """
    # Use default config if none provided
    if config is None:
        app_config = get_config()
        config = LLMConfig(
            temperature=app_config.llm_config.temperature,
            max_tokens=app_config.llm_config.max_tokens,
            top_p=app_config.llm_config.top_p,
        )

    # Step 1: Process contexts (deduplicate, rank, truncate if needed)
    processed_context = process_contexts(context, max_tokens=3000)

    # Step 2: Build structured prompt
    prompt = build_prompt(user_query, processed_context)

    # Step 3: Get appropriate provider
    provider = get_provider(model_name)

    # Step 4: Call LLM
    try:
        response = provider.generate(prompt, config)
        return response
    except LLMProviderError as e:
        # Re-raise provider errors
        raise e
    except Exception as e:
        raise LLMProviderError(f"Failed to generate response: {str(e)}")


def render_ui():
    """
    Main Streamlit UI loop for the Airline Insights Assistant.
    """
    # Page configuration
    st.set_page_config(
        page_title="Airline Insights Assistant",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "last_context" not in st.session_state:
        st.session_state.last_context = []
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    # Import retrieval modules
    from ingestion.main import main as process_user_query
    from retrieval.graph_retrieval import query_graph_cypher
    from retrieval.vector import query_graph_vector
    from frontend.config.models import get_all_model_names

    # Header
    st.title("✈️ Airline Flight Insights Assistant")
    st.markdown("*Powered by Knowledge Graph + LLM | Graph-RAG System*")
    st.divider()

    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        all_models = get_all_model_names()
        model_name = st.selectbox(
            "Select LLM Model",
            all_models,
            index=0,
            help="Choose which LLM model to use for generating responses",
        )

        # Retrieval method
        retrieval_method = st.radio(
            "Retrieval Method",
            ["Cypher Only", "Vector Only", "Both (Hybrid)"],
            index=2,
            help="Select how to retrieve context from the knowledge graph",
        )

        st.divider()

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            temperature = st.slider(
                "Temperature",
                0.0,
                1.0,
                0.7,
                0.1,
                help="Higher = more creative, Lower = more focused",
            )
            max_tokens = st.slider(
                "Max Tokens",
                256,
                4096,
                1024,
                256,
                help="Maximum length of the response",
            )
            top_p = st.slider(
                "Top P", 0.0, 1.0, 1.0, 0.05, help="Nucleus sampling parameter"
            )

        st.divider()

        # Session statistics
        st.subheader("📊 Session Stats")
        st.metric("Total Queries", len(st.session_state.query_history))
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")

        # API Key status
        st.divider()
        st.subheader("🔑 API Keys")
        from frontend.config.settings import validate_api_keys

        api_status = validate_api_keys()
        for provider, available in api_status.items():
            if available:
                st.success(f"{provider.capitalize()}: ✓")
            else:
                st.error(f"{provider.capitalize()}: ✗")

    # Main area: Two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        st.subheader("💬 Ask a Question")
        user_query = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the most delayed flights from ORD?\n"
            "e.g., Show me passenger satisfaction scores for flights to LHR\n"
            "e.g., Which aircraft types have the best food ratings?",
            height=100,
            key="user_query_input",
        )

        submit_button = st.button(
            "🔍 Submit Query", type="primary", use_container_width=True
        )

        # Process query
        if submit_button and user_query:
            with st.spinner("Processing your query..."):
                try:
                    # Step 1: Process input
                    processed_query = process_user_query(user_query)

                    # Step 2: Retrieve context
                    context_chunks = []

                    if retrieval_method in ["Cypher Only", "Both (Hybrid)"]:
                        with st.spinner("Querying graph database (Cypher)..."):
                            cypher_results = query_graph_cypher(processed_query)
                            context_chunks.extend(cypher_results)

                    if retrieval_method in ["Vector Only", "Both (Hybrid)"]:
                        with st.spinner("Performing vector search..."):
                            vector_results = query_graph_vector(user_query)
                            context_chunks.extend(vector_results)
                            print(
                                f"Vector search returned {len(vector_results)} results."
                            )

                    # Step 3: Generate response
                    with st.spinner(f"Generating response with {model_name}..."):
                        llm_config = LLMConfig(
                            temperature=temperature, max_tokens=max_tokens, top_p=top_p
                        )
                        llm_response = generate_response(
                            user_query,
                            context_chunks,
                            model_name=model_name,
                            config=llm_config,
                        )

                    # Store in session state
                    st.session_state.last_response = llm_response
                    st.session_state.last_context = context_chunks
                    st.session_state.total_cost += llm_response.cost_usd

                    # Add to history
                    st.session_state.query_history.append(
                        {
                            "query": user_query,
                            "response": llm_response,
                            "context": context_chunks,
                            "model": model_name,
                            "retrieval_method": retrieval_method,
                        }
                    )

                    st.success("Response generated successfully!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info(
                        "Please check your API keys in the sidebar or configuration file."
                    )

        # Display response
        if st.session_state.last_response:
            st.divider()
            st.subheader("🤖 Answer")
            st.markdown(st.session_state.last_response.text)

            # Metrics
            st.divider()
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Tokens", st.session_state.last_response.tokens_used)
            with metric_cols[1]:
                st.metric("Cost", f"${st.session_state.last_response.cost_usd:.4f}")
            with metric_cols[2]:
                st.metric(
                    "Time", f"{st.session_state.last_response.response_time_ms:.0f}ms"
                )
            with metric_cols[3]:
                st.metric(
                    "Model", st.session_state.last_response.model.split("-")[0].upper()
                )

    with col2:
        # Context display
        if st.session_state.last_context:
            st.subheader("📚 Retrieved Context")
            st.caption(f"Found {len(st.session_state.last_context)} relevant items")

            for i, ctx in enumerate(st.session_state.last_context[:10], 1):
                with st.expander(
                    f"#{i} [{ctx.source}] Score: {ctx.score:.2f}", expanded=(i <= 3)
                ):
                    st.write(ctx.text)
                    st.caption("Metadata:")
                    st.json(ctx.metadata)

    # Bottom tabs for advanced features
    st.divider()
    tab1, tab2, tab3 = st.tabs(
        ["📈 Graph Visualization", "🔄 Multi-Model Compare", "📜 Query History"]
    )

    with tab1:
        st.subheader("Knowledge Graph Visualization")
        if st.session_state.last_context:
            st.info(
                "Graph visualization feature coming soon. This will display the retrieved nodes and relationships from the knowledge graph."
            )
            # Placeholder for graph viz
        else:
            st.info("Submit a query to see the knowledge graph visualization.")

    with tab2:
        st.subheader("Multi-Model Comparison")
        if user_query:
            st.info(
                "Multi-model comparison feature coming soon. This will allow you to compare responses from multiple models side-by-side."
            )
            # Placeholder for multi-model comparison
        else:
            st.info("Enter a query above to compare responses across different models.")

    with tab3:
        st.subheader("Query History")
        if st.session_state.query_history:
            for i, item in enumerate(reversed(st.session_state.query_history), 1):
                with st.expander(
                    f"Query {len(st.session_state.query_history) - i + 1}: {item['query'][:50]}..."
                ):
                    st.write(f"**Query:** {item['query']}")
                    st.write(f"**Model:** {item['model']}")
                    st.write(f"**Retrieval:** {item['retrieval_method']}")
                    st.write(f"**Cost:** ${item['response'].cost_usd:.4f}")
                    st.write(f"**Response:** {item['response'].text[:200]}...")
        else:
            st.info("No queries yet. Start by asking a question above!")


if __name__ == "__main__":
    render_ui()

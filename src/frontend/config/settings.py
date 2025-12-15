import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""

    uri: str
    username: str
    password: str


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


@dataclass
class UIConfig:
    """UI-specific settings."""

    default_retrieval_method: str = "both"
    default_model: str = "gemini-1.5-flash"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_context_display: int = 10


@dataclass
class LLMConfigDefaults:
    """Default LLM generation parameters."""

    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0


@dataclass
class AppConfig:
    """Main application configuration."""

    neo4j: Neo4jConfig
    llm_providers: LLMProviderConfig
    ui_settings: UIConfig
    llm_config: LLMConfigDefaults


def _resolve_env_vars(value: str) -> str:
    """
    Resolve environment variable references in a string.

    Supports ${VAR_NAME} syntax.

    Args:
        value: String that may contain env var references

    Returns:
        String with env vars resolved
    """
    if not isinstance(value, str):
        return value

    # Find all ${VAR_NAME} patterns
    pattern = r"\$\{([^}]+)\}"
    matches = re.findall(pattern, value)

    for var_name in matches:
        env_value = os.getenv(var_name, "")
        value = value.replace(f"${{{var_name}}}", env_value)

    return value


def _load_yaml_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Recursively resolve environment variables
    def resolve_dict(d: dict) -> dict:
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = resolve_dict(value)
            elif isinstance(value, str):
                result[key] = _resolve_env_vars(value)
            else:
                result[key] = value
        return result

    return resolve_dict(config)


def load_config() -> AppConfig:
    """
    Load application configuration from multiple sources.

    Priority (highest to lowest):
    1. Environment variables
    2. config.yaml file
    3. Defaults

    Returns:
        AppConfig object

    Raises:
        ValueError: If critical configuration is missing
    """
    # Load environment variables from .env file (if exists)
    load_dotenv()

    # Try to load YAML config
    project_root = Path(__file__).parent.parent.parent.parent.parent
    config_path = project_root / "config.yaml"
    yaml_config = _load_yaml_config(config_path)

    # Neo4j Configuration
    neo4j_config = yaml_config.get("neo4j", {})
    neo4j = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", neo4j_config.get("uri", "bolt://localhost:7687")),
        username=os.getenv("NEO4J_USERNAME", neo4j_config.get("username", "neo4j")),
        password=os.getenv("NEO4J_PASSWORD", neo4j_config.get("password", "")),
    )

    # LLM Provider API Keys
    llm_providers_config = yaml_config.get("llm_providers", {})
    llm_providers = LLMProviderConfig(
        gemini_api_key=os.getenv(
            "GEMINI_API_KEY", llm_providers_config.get("gemini", {}).get("api_key", "")
        ),
        anthropic_api_key=os.getenv(
            "ANTHROPIC_API_KEY",
            llm_providers_config.get("claude", {}).get("api_key", ""),
        ),
        groq_api_key=os.getenv(
            "GROQ_API_KEY", llm_providers_config.get("groq", {}).get("api_key", "")
        ),
    )

    # UI Settings
    ui_config_dict = yaml_config.get("ui_settings", {})
    ui_settings = UIConfig(
        default_retrieval_method=os.getenv(
            "DEFAULT_RETRIEVAL_METHOD",
            ui_config_dict.get("default_retrieval_method", "both"),
        ),
        default_model=os.getenv(
            "DEFAULT_MODEL", ui_config_dict.get("default_model", "gemini-1.5-flash")
        ),
        cache_enabled=os.getenv(
            "CACHE_ENABLED", str(ui_config_dict.get("cache_enabled", True))
        ).lower()
        in ("true", "1", "yes"),
        cache_ttl_seconds=int(
            os.getenv(
                "CACHE_TTL_SECONDS", ui_config_dict.get("cache_ttl_seconds", 3600)
            )
        ),
        max_context_display=int(
            os.getenv(
                "MAX_CONTEXT_DISPLAY", ui_config_dict.get("max_context_display", 10)
            )
        ),
    )

    # LLM Config Defaults
    llm_config_dict = yaml_config.get("llm_config", {})
    llm_config = LLMConfigDefaults(
        temperature=float(
            os.getenv("LLM_TEMPERATURE", llm_config_dict.get("temperature", 0.7))
        ),
        max_tokens=int(
            os.getenv("LLM_MAX_TOKENS", llm_config_dict.get("max_tokens", 1024))
        ),
        top_p=float(os.getenv("LLM_TOP_P", llm_config_dict.get("top_p", 1.0))),
    )

    return AppConfig(
        neo4j=neo4j,
        llm_providers=llm_providers,
        ui_settings=ui_settings,
        llm_config=llm_config,
    )


# Singleton instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the application configuration (singleton).

    Returns:
        AppConfig object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def validate_api_keys() -> dict[str, bool]:
    """
    Check which API keys are configured.

    Returns:
        Dictionary mapping provider to availability status
    """
    config = get_config()
    return {
        "gemini": bool(config.llm_providers.gemini_api_key),
        "claude": bool(config.llm_providers.anthropic_api_key),
        "groq": bool(config.llm_providers.groq_api_key),
    }


def get_missing_api_keys() -> list[str]:
    """
    Get list of providers with missing API keys.

    Returns:
        List of provider names
    """
    availability = validate_api_keys()
    return [provider for provider, available in availability.items() if not available]

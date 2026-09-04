"""
Global configuration loading for ATAR via pydantic-settings.
"""
from typing import Annotated, Literal
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass

class AtarConfig(BaseSettings):
    """
    Global configuration for the ATAR project.
    Values can be set via environment variables (e.g., ATAR_RANDOM_SEED)
    or in a .env file.
    """
    # Required parameters (no defaults)
    data_dir: str = Field(description="Base directory for data")
    
    # Optional parameters with defaults
    model_dir: str = Field(default="models", description="Base directory for model checkpoints")
    random_seed: int = Field(default=42, description="Global random seed")
    device: str = Field(default="cpu", description="Compute device (e.g., 'cpu', 'cuda')")
    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Field(description="Logging verbosity level")
    ] = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="ATAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def load_config() -> AtarConfig:
    try:
        return AtarConfig()
    except ValidationError as e:
        missing_vars = []
        for error in e.errors():
            if error["type"] == "missing":
                missing_vars.append(error["loc"][0])
        
        if missing_vars:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing_vars)}") from e
        raise ConfigError(f"Configuration error: {e}") from e

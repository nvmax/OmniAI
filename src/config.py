"""
Configuration module for Omni-Assistant Discord Bot.
Handles environment variables and application settings.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field

# Load environment variables
load_dotenv()

class Config(BaseSettings):
    """Application configuration using Pydantic BaseSettings."""
    
    # Discord Configuration
    discord_bot_token: str = Field(..., env="DISCORD_BOT_TOKEN")
    discord_server_id: Optional[str] = Field(None, env="DISCORD_SERVER_ID")
    discord_channel_ids: List[str] = Field(
        default_factory=lambda: os.getenv("DISCORD_CHANNEL_ID", "").split(",") if os.getenv("DISCORD_CHANNEL_ID") else []
    )
    
    # LLM Provider Configuration
    llm_provider: str = Field("lm_studio", env="LLM_PROVIDER")  # "lm_studio", "gemini", or "auto"

    # LM Studio Configuration
    lm_studio_host: str = Field("127.0.0.1", env="LM_STUDIO_HOST")
    lm_studio_port: int = Field(1234, env="LM_STUDIO_PORT")
    lm_studio_model: str = Field("local-model", env="LM_STUDIO_MODEL")

    # Google Gemini Configuration
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", env="GEMINI_MODEL")  # or "gemini-1.5-pro"
    gemini_temperature: float = Field(0.7, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(8192, env="GEMINI_MAX_TOKENS")
    
    # Vector Database Configuration
    vector_db_path: str = Field("./data/vector_db", env="VECTOR_DB_PATH")
    
    # Bot Configuration
    default_bot_personality: str = Field("conversational", env="DEFAULT_BOT_PERSONALITY")
    bot_name: str = Field("Omni-Assistant", env="BOT_NAME")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/bot.log", env="LOG_FILE")
    
    # Memory Configuration
    max_short_term_memory: int = Field(50, env="MAX_SHORT_TERM_MEMORY")
    max_context_retrieval: int = Field(5, env="MAX_CONTEXT_RETRIEVAL")
    memory_cleanup_interval: int = Field(3600, env="MEMORY_CLEANUP_INTERVAL")

    # LLM Token Limits Configuration
    default_max_tokens: int = Field(4000, env="DEFAULT_MAX_TOKENS")
    research_max_tokens: int = Field(6000, env="RESEARCH_MAX_TOKENS")
    coding_max_tokens: int = Field(5000, env="CODING_MAX_TOKENS")
    general_max_tokens: int = Field(4000, env="GENERAL_MAX_TOKENS")


    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields in environment

    @property
    def lm_studio_url(self) -> str:
        """Get the complete LM Studio API URL."""
        return f"http://{self.lm_studio_host}:{self.lm_studio_port}"
    
    @property
    def lm_studio_chat_url(self) -> str:
        """Get the LM Studio chat completions endpoint URL."""
        return f"{self.lm_studio_url}/v1/chat/completions"
    
    def is_valid_channel(self, channel_id: str) -> bool:
        """Check if a channel ID is in the configured list."""
        return not self.discord_channel_ids or channel_id in self.discord_channel_ids
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            os.path.dirname(self.vector_db_path),
            os.path.dirname(self.log_file),
            "data",
            "logs"
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config()

# Ensure directories exist on import
config.ensure_directories()

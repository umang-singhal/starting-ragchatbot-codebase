import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    enabled: bool = True
    level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    format: str = os.getenv(
        "LOG_FORMAT",
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    date_format: str = '%Y-%m-%d %H:%M:%S'


@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "glm-4.7")

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "2"))  # Maximum sequential tool calls per query

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)


config = Config()



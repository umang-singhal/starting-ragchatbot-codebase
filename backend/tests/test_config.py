"""
Tests for config.py - Configuration validation.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


@pytest.mark.unit
class TestConfig:
    """Tests for Config class."""

    def test_config_max_results_should_be_positive(self):
        """
        CRITICAL TEST: MAX_RESULTS must be > 0 for searches to return results.

        This test will FAIL with the current bug (MAX_RESULTS=0) and PASS after the fix.
        """
        config = Config()

        # This assertion will FAIL with the bug
        assert config.MAX_RESULTS > 0, (
            "MAX_RESULTS=0 causes all vector searches to return 0 results. "
            "This is the root cause of 'query failed' errors."
        )

    def test_config_chunk_size_should_be_reasonable(self):
        """CHUNK_SIZE should be between 100 and 2000."""
        config = Config()
        assert (
            100 <= config.CHUNK_SIZE <= 2000
        ), f"CHUNK_SIZE={config.CHUNK_SIZE} is outside reasonable range [100, 2000]"

    def test_config_chunk_overlap_less_than_chunk_size(self):
        """CHUNK_OVERLAP should be less than CHUNK_SIZE."""
        config = Config()
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, (
            f"CHUNK_OVERLAP={config.CHUNK_OVERLAP} should be less than "
            f"CHUNK_SIZE={config.CHUNK_SIZE}"
        )

    def test_config_max_history_positive(self):
        """MAX_HISTORY must be positive."""
        config = Config()
        assert config.MAX_HISTORY > 0, f"MAX_HISTORY={config.MAX_HISTORY} should be positive"

    def test_config_has_embedding_model(self):
        """EMBEDDING_MODEL should be set."""
        config = Config()
        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL should be set"
        assert isinstance(config.EMBEDDING_MODEL, str)

    def test_config_has_chroma_path(self):
        """CHROMA_PATH should be set."""
        config = Config()
        assert config.CHROMA_PATH, "CHROMA_PATH should be set"
        assert isinstance(config.CHROMA_PATH, str)

    def test_config_has_anthropic_model(self):
        """ANTHROPIC_MODEL should be set."""
        config = Config()
        assert config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL should be set"
        assert isinstance(config.ANTHROPIC_MODEL, str)

    def test_config_logging_config(self):
        """Logging configuration should be properly initialized."""
        config = Config()
        assert config.logging is not None
        assert hasattr(config.logging, "level")
        assert hasattr(config.logging, "format")

    def test_config_logging_level_valid(self):
        """Logging level should be a valid Python logging level."""
        config = Config()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert (
            config.logging.level in valid_levels
        ), f"LOG_LEVEL={config.logging.level} is not a valid logging level"

    def test_config_defaults(self):
        """Test that default configuration values are set correctly."""
        config = Config()

        # Check defaults (before any bug fix)
        assert config.CHUNK_SIZE == 800
        assert config.CHUNK_OVERLAP == 100
        # MAX_RESULTS will be 0 (bug) or 5 (fixed)
        assert config.MAX_HISTORY == 2
        assert config.CHROMA_PATH == "./chroma_db"

    def test_config_bug_report(self):
        """
        Diagnostic test that reports the current state of MAX_RESULTS.

        This test always passes but provides diagnostic information.
        """
        config = Config()

        if config.MAX_RESULTS == 0:
            pytest.fail(
                "BUG DETECTED: MAX_RESULTS=0 in config.py:34\n"
                "This causes ChromaDB queries to return 0 results.\n"
                "Fix: Change MAX_RESULTS: int = 0 to MAX_RESULTS: int = 5"
            )
        else:
            # Bug is fixed
            assert config.MAX_RESULTS > 0, "MAX_RESULTS should be positive"

    def test_config_impact_analysis(self):
        """
        Test that demonstrates the impact of MAX_RESULTS=0.

        This test explains how the bug affects the system.
        """
        config = Config()

        # The bug impact chain:
        # 1. config.MAX_RESULTS = 0
        # 2. VectorStore.__init__ sets self.max_results = 0
        # 3. VectorStore.search() uses search_limit = self.max_results
        # 4. ChromaDB query(n_results=0) returns empty results
        # 5. CourseSearchTool.execute() receives empty results
        # 6. Returns "No relevant content found"
        # 7. User sees "query failed"

        if config.MAX_RESULTS == 0:
            pytest.fail(
                "BUG IMPACT:\n"
                "1. Config MAX_RESULTS=0 → VectorStore.max_results=0\n"
                "2. VectorStore.search() passes n_results=0 to ChromaDB\n"
                "3. ChromaDB returns empty results\n"
                "4. CourseSearchTool returns 'No relevant content found'\n"
                "5. User experiences 'query failed' for all content questions\n\n"
                "FIX: Change line 34 in config.py from:\n"
                "    MAX_RESULTS: int = 0\n"
                "to:\n"
                "    MAX_RESULTS: int = 5"
            )

        assert True  # Pass if bug is fixed

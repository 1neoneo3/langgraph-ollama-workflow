"""Tests for configuration settings."""

import pytest
from src.config.settings import Config


class TestConfig:
    """Test cases for Config class."""

    def test_ollama_settings_exist(self):
        """Test that Ollama settings are defined."""
        assert hasattr(Config, 'OLLAMA_MODEL')
        assert hasattr(Config, 'OLLAMA_BASE_URL')
        assert hasattr(Config, 'OLLAMA_TEMPERATURE')
        
        assert isinstance(Config.OLLAMA_MODEL, str)
        assert isinstance(Config.OLLAMA_BASE_URL, str)
        assert isinstance(Config.OLLAMA_TEMPERATURE, (int, float))
        
        assert len(Config.OLLAMA_MODEL) > 0
        assert Config.OLLAMA_BASE_URL.startswith('http')
        assert 0 <= Config.OLLAMA_TEMPERATURE <= 1

    def test_search_settings_exist(self):
        """Test that search settings are defined."""
        assert hasattr(Config, 'DEFAULT_SEARCH_DAYS_LIMIT')
        assert hasattr(Config, 'SEARCH_RESULT_LIMIT')
        assert hasattr(Config, 'PARALLEL_SEARCH_LIMIT')
        assert hasattr(Config, 'SEARCH_TIMEOUT')
        assert hasattr(Config, 'INDIVIDUAL_RESULT_LIMIT')
        
        # Check types
        assert isinstance(Config.DEFAULT_SEARCH_DAYS_LIMIT, int)
        assert isinstance(Config.SEARCH_RESULT_LIMIT, int)
        assert isinstance(Config.PARALLEL_SEARCH_LIMIT, int)
        assert isinstance(Config.SEARCH_TIMEOUT, int)
        assert isinstance(Config.INDIVIDUAL_RESULT_LIMIT, int)
        
        # Check reasonable values
        assert Config.DEFAULT_SEARCH_DAYS_LIMIT > 0
        assert Config.SEARCH_RESULT_LIMIT > 0
        assert Config.PARALLEL_SEARCH_LIMIT > 0
        assert Config.SEARCH_TIMEOUT > 0
        assert Config.INDIVIDUAL_RESULT_LIMIT > 0

    def test_time_descriptions_structure(self):
        """Test TIME_DESCRIPTIONS structure."""
        assert hasattr(Config, 'TIME_DESCRIPTIONS')
        assert isinstance(Config.TIME_DESCRIPTIONS, dict)
        
        # Check that all keys are integers and values are strings
        for key, value in Config.TIME_DESCRIPTIONS.items():
            assert isinstance(key, int)
            assert isinstance(value, str)
            assert key > 0
            assert len(value) > 0
            assert "過去" in value

    def test_time_descriptions_content(self):
        """Test specific TIME_DESCRIPTIONS content."""
        descriptions = Config.TIME_DESCRIPTIONS
        
        # Check specific known values
        assert 1 in descriptions
        assert 7 in descriptions
        assert 30 in descriptions
        assert 365 in descriptions
        
        assert descriptions[1] == "過去1日"
        assert descriptions[7] == "過去1週間"
        assert descriptions[30] == "過去1ヶ月"
        assert descriptions[365] == "過去1年"

    def test_recent_keywords_structure(self):
        """Test RECENT_KEYWORDS structure."""
        assert hasattr(Config, 'RECENT_KEYWORDS')
        assert isinstance(Config.RECENT_KEYWORDS, list)
        assert len(Config.RECENT_KEYWORDS) > 0
        
        # Check all items are strings
        for keyword in Config.RECENT_KEYWORDS:
            assert isinstance(keyword, str)
            assert len(keyword) > 0

    def test_recent_keywords_content(self):
        """Test specific RECENT_KEYWORDS content."""
        keywords = Config.RECENT_KEYWORDS
        
        # Check for Japanese keywords
        assert "最新" in keywords
        assert "最近" in keywords
        assert "今日" in keywords
        assert "今週" in keywords
        assert "今月" in keywords
        
        # Check for English keywords
        assert "latest" in keywords
        assert "recent" in keywords
        assert "today" in keywords
        assert "current" in keywords

    def test_time_specific_keywords_structure(self):
        """Test TIME_SPECIFIC_KEYWORDS structure."""
        assert hasattr(Config, 'TIME_SPECIFIC_KEYWORDS')
        assert isinstance(Config.TIME_SPECIFIC_KEYWORDS, dict)
        assert len(Config.TIME_SPECIFIC_KEYWORDS) > 0
        
        # Check structure
        for keyword, days in Config.TIME_SPECIFIC_KEYWORDS.items():
            assert isinstance(keyword, str)
            assert isinstance(days, int)
            assert len(keyword) > 0
            assert days > 0

    def test_time_specific_keywords_content(self):
        """Test specific TIME_SPECIFIC_KEYWORDS content."""
        keywords = Config.TIME_SPECIFIC_KEYWORDS
        
        # Check specific mappings
        assert keywords["今日"] == 1
        assert keywords["today"] == 1
        assert keywords["今週"] == 7
        assert keywords["this week"] == 7
        assert keywords["今月"] == 30
        assert keywords["this month"] == 30

    def test_slack_settings_exist(self):
        """Test that Slack settings are defined."""
        assert hasattr(Config, 'SLACK_MAX_RETRIES')
        assert hasattr(Config, 'SLACK_INITIAL_RETRY_DELAY')
        assert hasattr(Config, 'SLACK_CONTENT_LIMIT')
        
        # Check types
        assert isinstance(Config.SLACK_MAX_RETRIES, int)
        assert isinstance(Config.SLACK_INITIAL_RETRY_DELAY, int)
        assert isinstance(Config.SLACK_CONTENT_LIMIT, int)
        
        # Check reasonable values
        assert Config.SLACK_MAX_RETRIES > 0
        assert Config.SLACK_INITIAL_RETRY_DELAY > 0
        assert Config.SLACK_CONTENT_LIMIT > 0

    def test_claude_settings_exist(self):
        """Test that Claude Code settings are defined."""
        assert hasattr(Config, 'CLAUDE_MAX_TURNS')
        assert hasattr(Config, 'CLAUDE_WEBSEARCH_MAX_TURNS')
        
        # Check types
        assert isinstance(Config.CLAUDE_MAX_TURNS, int)
        assert isinstance(Config.CLAUDE_WEBSEARCH_MAX_TURNS, int)
        
        # Check reasonable values
        assert Config.CLAUDE_MAX_TURNS > 0
        assert Config.CLAUDE_WEBSEARCH_MAX_TURNS > 0

    def test_threading_settings_exist(self):
        """Test that threading settings are defined."""
        assert hasattr(Config, 'MAX_WORKERS')
        
        # Check type
        assert isinstance(Config.MAX_WORKERS, int)
        
        # Check reasonable value
        assert Config.MAX_WORKERS > 0

    def test_config_class_is_not_instantiable(self):
        """Test that Config is used as a class with class variables."""
        # This tests that Config is meant to be used as a container of constants
        # rather than being instantiated
        
        # All attributes should be accessible without instantiation
        assert Config.OLLAMA_MODEL is not None
        assert Config.SEARCH_RESULT_LIMIT is not None
        assert Config.TIME_DESCRIPTIONS is not None
        
        # Config can be instantiated but it's not necessary
        config_instance = Config()
        assert config_instance.OLLAMA_MODEL == Config.OLLAMA_MODEL

    def test_all_required_constants_exist(self):
        """Test that all expected configuration constants exist."""
        required_constants = [
            'OLLAMA_MODEL', 'OLLAMA_BASE_URL', 'OLLAMA_TEMPERATURE',
            'DEFAULT_SEARCH_DAYS_LIMIT', 'SEARCH_RESULT_LIMIT', 
            'PARALLEL_SEARCH_LIMIT', 'SEARCH_TIMEOUT', 'INDIVIDUAL_RESULT_LIMIT',
            'TIME_DESCRIPTIONS', 'RECENT_KEYWORDS', 'TIME_SPECIFIC_KEYWORDS',
            'SLACK_MAX_RETRIES', 'SLACK_INITIAL_RETRY_DELAY', 'SLACK_CONTENT_LIMIT',
            'CLAUDE_MAX_TURNS', 'CLAUDE_WEBSEARCH_MAX_TURNS',
            'MAX_WORKERS'
        ]
        
        for constant in required_constants:
            assert hasattr(Config, constant), f"Missing constant: {constant}"

    def test_config_values_consistency(self):
        """Test consistency between related configuration values."""
        # INDIVIDUAL_RESULT_LIMIT should be less than SEARCH_RESULT_LIMIT
        assert Config.INDIVIDUAL_RESULT_LIMIT <= Config.SEARCH_RESULT_LIMIT
        
        # MAX_WORKERS should be reasonable
        assert Config.MAX_WORKERS <= 10  # Reasonable upper limit
        
        # Timeout should be longer than retry delay
        assert Config.SEARCH_TIMEOUT > Config.SLACK_INITIAL_RETRY_DELAY
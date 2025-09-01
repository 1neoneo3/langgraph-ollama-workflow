"""Configuration settings for the LangGraph workflow application."""

from typing import Dict, List


class Config:
    """Application configuration constants."""
    
    # Ollama settings
    OLLAMA_MODEL = "gpt-oss:20b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_TEMPERATURE = 0.7

    # Search settings
    DEFAULT_SEARCH_DAYS_LIMIT = 60
    SEARCH_RESULT_LIMIT = 2000
    PARALLEL_SEARCH_LIMIT = 3
    SEARCH_TIMEOUT = 120
    INDIVIDUAL_RESULT_LIMIT = 1000

    # Time descriptions mapping
    TIME_DESCRIPTIONS = {
        1: "過去1日",
        7: "過去1週間",
        30: "過去1ヶ月",
        60: "過去2ヶ月",
        90: "過去3ヶ月",
        180: "過去6ヶ月",
        365: "過去1年",
    }

    # Recent search keywords
    RECENT_KEYWORDS = [
        "最新",
        "直近",
        "最近",
        "新しい",
        "今日",
        "今週",
        "今月",
        "latest",
        "recent",
        "new",
        "current",
        "today",
        "this week",
        "this month",
        "今年",
        "this year",
        "最新版",
        "最新バージョン",
        "current version",
        "latest version",
        "up to date",
        "アップデート",
        "update",
    ]

    # Time-specific keywords mapping
    TIME_SPECIFIC_KEYWORDS = {
        "今日": 1,
        "today": 1,
        "今週": 7,
        "this week": 7,
        "今月": 30,
        "this month": 30,
        "直近": 60,
        "最近": 60,
        "recent": 60,
    }

    # Slack settings
    SLACK_MAX_RETRIES = 3
    SLACK_INITIAL_RETRY_DELAY = 2
    SLACK_CONTENT_LIMIT = 3000

    # Claude Code settings
    CLAUDE_MAX_TURNS = 1
    CLAUDE_WEBSEARCH_MAX_TURNS = 3

    # Threading settings
    MAX_WORKERS = 3
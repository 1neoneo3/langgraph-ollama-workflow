"""Tests for helper functions."""

import pytest
from src.utils.helpers import (
    create_system_prompt,
    build_psearch_command,
    format_parallel_search_results
)


class TestHelpers:
    """Test cases for helper functions."""

    def test_create_system_prompt(self):
        """Test system prompt creation."""
        content = "テスト質問"
        current_date_info = {
            "date_str": "2024年09月02日",
            "year": 2024
        }
        search_results = "検索結果のサンプル"
        iteration = 1
        
        prompt = create_system_prompt(content, current_date_info, search_results, iteration)
        
        # Check prompt contains expected elements
        assert isinstance(prompt, str)
        assert content in prompt
        assert current_date_info["date_str"] in prompt
        assert str(current_date_info["year"]) in prompt
        assert search_results in prompt
        assert str(iteration) in prompt
        assert "日本語" in prompt
        assert "LangGraph" in prompt

    def test_create_system_prompt_no_search_results(self):
        """Test system prompt creation without search results."""
        content = "テスト質問"
        current_date_info = {
            "date_str": "2024年09月02日",
            "year": 2024
        }
        search_results = ""
        iteration = 2
        
        prompt = create_system_prompt(content, current_date_info, search_results, iteration)
        
        assert "検索結果がありません" in prompt
        assert str(iteration) in prompt

    def test_build_psearch_command_basic(self):
        """Test basic psearch command building."""
        query = "test query"
        recent_search_mode = False
        search_days_limit = 30
        
        cmd = build_psearch_command(query, recent_search_mode, search_days_limit)
        
        assert isinstance(cmd, list)
        assert "psearch" in cmd
        assert "search" in cmd
        assert query in cmd
        assert "-n" in cmd
        assert "5" in cmd
        assert "-c" in cmd
        assert "--json" in cmd

    def test_build_psearch_command_recent_mode_short(self):
        """Test psearch command with recent mode (short period)."""
        query = "recent query"
        recent_search_mode = True
        search_days_limit = 7  # Short period
        
        cmd = build_psearch_command(query, recent_search_mode, search_days_limit)
        
        assert "-r" in cmd
        assert "-s" in cmd

    def test_build_psearch_command_recent_mode_long(self):
        """Test psearch command with recent mode (long period)."""
        query = "older query"
        recent_search_mode = True
        search_days_limit = 90  # Long period
        
        cmd = build_psearch_command(query, recent_search_mode, search_days_limit)
        
        assert "-r" in cmd
        assert "--months" in cmd
        assert "3" in cmd  # 90 days ≈ 3 months

    def test_build_psearch_command_long_query(self):
        """Test psearch command with long query (should be truncated)."""
        long_query = "a" * 200  # 200 characters
        recent_search_mode = False
        search_days_limit = 30
        
        cmd = build_psearch_command(long_query, recent_search_mode, search_days_limit)
        
        # Find the query in command
        query_in_cmd = None
        for item in cmd:
            if "a" in item:
                query_in_cmd = item
                break
        
        assert query_in_cmd is not None
        assert len(query_in_cmd) <= 100

    def test_format_parallel_search_results_successful(self):
        """Test formatting successful parallel search results."""
        search_results = [
            {
                "query": "test query 1",
                "success": True,
                "results": "Result content 1",
                "elapsed_time": 1.5
            },
            {
                "query": "test query 2",
                "success": True,
                "results": "Result content 2",
                "elapsed_time": 2.0
            }
        ]
        total_elapsed_time = 2.5
        
        formatted = format_parallel_search_results(search_results, total_elapsed_time)
        
        assert isinstance(formatted, str)
        assert "Parallel Search Results" in formatted
        assert "2/2 successful" in formatted
        assert "✅" in formatted
        assert "test query 1" in formatted
        assert "test query 2" in formatted
        assert "1.50s" in formatted
        assert "2.00s" in formatted
        assert "Result content 1" in formatted
        assert "Result content 2" in formatted

    def test_format_parallel_search_results_mixed_success(self):
        """Test formatting parallel search results with mixed success/failure."""
        search_results = [
            {
                "query": "successful query",
                "success": True,
                "results": "Good result",
                "elapsed_time": 1.0
            },
            {
                "query": "failed query",
                "success": False,
                "results": "Error message",
                "elapsed_time": 0.5
            }
        ]
        total_elapsed_time = 1.5
        
        formatted = format_parallel_search_results(search_results, total_elapsed_time)
        
        assert "1/2 successful" in formatted
        assert "✅" in formatted
        assert "❌" in formatted
        assert "successful query" in formatted
        assert "failed query" in formatted
        assert "Good result" in formatted
        assert "Error: Error message" in formatted

    def test_format_parallel_search_results_empty(self):
        """Test formatting empty search results."""
        search_results = []
        total_elapsed_time = 0.0
        
        formatted = format_parallel_search_results(search_results, total_elapsed_time)
        
        assert "0/0 successful" in formatted
        assert isinstance(formatted, str)

    def test_format_parallel_search_results_no_results_content(self):
        """Test formatting search results with no content."""
        search_results = [
            {
                "query": "empty query",
                "success": True,
                "results": "",
                "elapsed_time": 1.0
            }
        ]
        total_elapsed_time = 1.0
        
        formatted = format_parallel_search_results(search_results, total_elapsed_time)
        
        assert "1/1 successful" in formatted
        assert "empty query" in formatted
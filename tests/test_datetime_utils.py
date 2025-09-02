"""Tests for datetime utility functions."""

import pytest
from datetime import datetime
from src.utils.datetime_utils import (
    get_current_datetime_info,
    get_time_description,
    detect_recent_search_mode
)


class TestDateTimeUtils:
    """Test cases for datetime utility functions."""

    def test_get_current_datetime_info(self):
        """Test current datetime info function returns expected structure."""
        result = get_current_datetime_info()
        
        # Check required keys exist
        assert "datetime" in result
        assert "year" in result
        assert "month" in result
        assert "day" in result
        assert "date_str" in result
        
        # Check types
        assert isinstance(result["datetime"], datetime)
        assert isinstance(result["year"], int)
        assert isinstance(result["month"], int)
        assert isinstance(result["day"], int)
        assert isinstance(result["date_str"], str)
        
        # Check values are reasonable
        assert 2020 <= result["year"] <= 3000
        assert 1 <= result["month"] <= 12
        assert 1 <= result["day"] <= 31
        assert f"{result['year']}年{result['month']:02d}月{result['day']:02d}日" == result["date_str"]

    def test_get_time_description(self):
        """Test time description function."""
        # Test known values
        assert get_time_description(1) == "過去1日"
        assert get_time_description(7) == "過去1週間"
        assert get_time_description(30) == "過去1ヶ月"
        assert get_time_description(365) == "過去1年"
        
        # Test unknown value falls back to default format
        assert get_time_description(100) == "過去100日"

    def test_detect_recent_search_mode_basic(self):
        """Test basic recent search mode detection."""
        current_date_info = {
            "year": 2024,
            "month": 9,
            "day": 2,
            "date_str": "2024年09月02日"
        }
        
        # Test with recent keywords
        recent_mode, days = detect_recent_search_mode("最新の情報を教えて", current_date_info)
        assert recent_mode is True
        assert isinstance(days, int)
        
        # Test without recent keywords
        recent_mode, days = detect_recent_search_mode("普通の質問", current_date_info)
        assert recent_mode is False
        assert isinstance(days, int)

    def test_detect_recent_search_mode_with_year(self):
        """Test recent search mode with year keywords."""
        current_date_info = {
            "year": 2024,
            "month": 9,
            "day": 2,
            "date_str": "2024年09月02日"
        }
        
        # Test with current year
        recent_mode, days = detect_recent_search_mode("2024年のトレンド", current_date_info)
        assert recent_mode is True
        
        # Test with previous year
        recent_mode, days = detect_recent_search_mode("2023年の情報", current_date_info)
        assert recent_mode is True

    def test_detect_recent_search_mode_time_limits(self):
        """Test recent search mode with specific time limits."""
        current_date_info = {
            "year": 2024,
            "month": 9,
            "day": 2,
            "date_str": "2024年09月02日"
        }
        
        # Test with "今日" keyword should return shorter time limit
        recent_mode, days = detect_recent_search_mode("今日の最新情報", current_date_info)
        assert recent_mode is True
        assert days <= 7  # Should be short time limit
        
        # Test with "最新" keyword
        recent_mode, days = detect_recent_search_mode("最新のニュース", current_date_info)
        assert recent_mode is True
        assert days > 0

    @pytest.mark.parametrize("keyword", [
        "最新", "新しい", "最近", "今年", "今日", "今週", "今月"
    ])
    def test_recent_keywords_detection(self, keyword):
        """Test various recent keywords are detected."""
        current_date_info = {
            "year": 2024,
            "month": 9,
            "day": 2,
            "date_str": "2024年09月02日"
        }
        
        text = f"{keyword}の情報を教えて"
        recent_mode, days = detect_recent_search_mode(text, current_date_info)
        assert recent_mode is True

    def test_detect_recent_search_mode_return_types(self):
        """Test function returns correct types."""
        current_date_info = {
            "year": 2024,
            "month": 9,
            "day": 2,
            "date_str": "2024年09月02日"
        }
        
        result = detect_recent_search_mode("テスト", current_date_info)
        
        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        # Check types
        recent_mode, days = result
        assert isinstance(recent_mode, bool)
        assert isinstance(days, int)
        assert days > 0
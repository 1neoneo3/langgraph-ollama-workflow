"""Date and time utility functions."""

import datetime
from typing import Dict
from ..config.settings import Config


def get_current_datetime_info() -> Dict[str, any]:
    """Get current datetime information in a consistent format."""
    current_datetime = datetime.datetime.now()
    return {
        "datetime": current_datetime,
        "year": current_datetime.year,
        "month": current_datetime.month,
        "day": current_datetime.day,
        "date_str": current_datetime.strftime("%Y年%m月%d日"),
    }


def get_time_description(days: int) -> str:
    """Get human-readable time description for given days."""
    return Config.TIME_DESCRIPTIONS.get(days, f"過去{days}日")


def detect_recent_search_mode(user_input: str, current_date_info: Dict[str, any]) -> tuple[bool, int]:
    """Detect if recent search mode should be activated and determine time limit."""
    # Enhanced keywords including dynamic current year
    recent_keywords = Config.RECENT_KEYWORDS + [
        f"{current_date_info['year']}年",
        f"{current_date_info['year'] - 1}年",
    ]

    recent_search_mode = any(keyword in user_input for keyword in recent_keywords)

    # Determine specific time range
    search_days_limit = Config.DEFAULT_SEARCH_DAYS_LIMIT
    for keyword, days in Config.TIME_SPECIFIC_KEYWORDS.items():
        if keyword in user_input:
            search_days_limit = min(search_days_limit, days)
            break

    if recent_search_mode:
        time_description = get_time_description(search_days_limit)
        print(
            f"🔍 Recent information keywords detected - search will be limited to {time_description}"
        )
        filter_year = current_date_info["year"] - (1 if search_days_limit > 30 else 0)
        print(
            f"📅 Current date: {current_date_info['date_str']} - filtering for content from {filter_year}年以降"
        )

    return recent_search_mode, search_days_limit
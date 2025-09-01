"""Core workflow state definitions."""

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from typing import List, Dict


class WorkflowState(TypedDict):
    """State structure for the LangGraph workflow."""
    
    messages: list[BaseMessage]
    iteration: int
    user_input: str
    original_user_input: str  # Store original question for iterations
    processed_output: str
    should_continue: bool
    search_results: str
    search_queries: list[str]  # Store generated search queries
    parallel_search_stats: dict  # Store parallel search statistics
    recent_search_mode: bool
    search_days_limit: int  # Store specific time limit for search filtering
    initial_output: str  # Store first AI output for comparison
    reviewed_output: str  # Store Claude Code reviewed output
    document_generated: bool  # Track document generation status
    document_content: str  # Store generated markdown content
    document_path: str  # Store path to generated document
    slack_notification_sent: bool  # Track Slack notification status
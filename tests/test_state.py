"""Tests for workflow state."""

import pytest
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from src.core.state import WorkflowState


class TestWorkflowState:
    """Test cases for WorkflowState."""

    def test_workflow_state_structure(self):
        """Test WorkflowState has all required fields."""
        # Create a sample state
        state = WorkflowState(
            messages=[HumanMessage(content="test")],
            iteration=1,
            user_input="test input",
            original_user_input="original test",
            processed_output="processed",
            should_continue=True,
            search_results="results",
            search_queries=["query1", "query2"],
            parallel_search_stats={"count": 2},
            recent_search_mode=False,
            search_days_limit=30,
            initial_output="initial",
            reviewed_output="reviewed",
            document_generated=False,
            document_content="",
            document_path="",
            slack_notification_sent=False
        )
        
        # Check all required fields exist
        assert "messages" in state
        assert "iteration" in state
        assert "user_input" in state
        assert "original_user_input" in state
        assert "processed_output" in state
        assert "should_continue" in state
        assert "search_results" in state
        assert "search_queries" in state
        assert "parallel_search_stats" in state
        assert "recent_search_mode" in state
        assert "search_days_limit" in state
        assert "initial_output" in state
        assert "reviewed_output" in state
        assert "document_generated" in state
        assert "document_content" in state
        assert "document_path" in state
        assert "slack_notification_sent" in state

    def test_workflow_state_types(self):
        """Test WorkflowState field types."""
        messages = [HumanMessage(content="test"), AIMessage(content="response")]
        search_queries = ["query1", "query2"]
        parallel_search_stats = {"successful": 2, "failed": 0}
        
        state = WorkflowState(
            messages=messages,
            iteration=1,
            user_input="test input",
            original_user_input="original test",
            processed_output="processed",
            should_continue=True,
            search_results="results",
            search_queries=search_queries,
            parallel_search_stats=parallel_search_stats,
            recent_search_mode=False,
            search_days_limit=30,
            initial_output="initial",
            reviewed_output="reviewed",
            document_generated=False,
            document_content="document content",
            document_path="/path/to/doc",
            slack_notification_sent=True
        )
        
        # Check types
        assert isinstance(state["messages"], list)
        assert isinstance(state["iteration"], int)
        assert isinstance(state["user_input"], str)
        assert isinstance(state["original_user_input"], str)
        assert isinstance(state["processed_output"], str)
        assert isinstance(state["should_continue"], bool)
        assert isinstance(state["search_results"], str)
        assert isinstance(state["search_queries"], list)
        assert isinstance(state["parallel_search_stats"], dict)
        assert isinstance(state["recent_search_mode"], bool)
        assert isinstance(state["search_days_limit"], int)
        assert isinstance(state["initial_output"], str)
        assert isinstance(state["reviewed_output"], str)
        assert isinstance(state["document_generated"], bool)
        assert isinstance(state["document_content"], str)
        assert isinstance(state["document_path"], str)
        assert isinstance(state["slack_notification_sent"], bool)

    def test_workflow_state_message_types(self):
        """Test WorkflowState with different message types."""
        messages = [
            HumanMessage(content="Human message"),
            AIMessage(content="AI response"),
        ]
        
        state = WorkflowState(
            messages=messages,
            iteration=1,
            user_input="test",
            original_user_input="test",
            processed_output="output",
            should_continue=False,
            search_results="",
            search_queries=[],
            parallel_search_stats={},
            recent_search_mode=True,
            search_days_limit=7,
            initial_output="",
            reviewed_output="",
            document_generated=False,
            document_content="",
            document_path="",
            slack_notification_sent=False
        )
        
        # Check message types
        assert len(state["messages"]) == 2
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
        assert state["messages"][0].content == "Human message"
        assert state["messages"][1].content == "AI response"

    def test_workflow_state_search_queries(self):
        """Test WorkflowState with search queries."""
        search_queries = ["Python tutorial", "FastAPI documentation", "pytest guide"]
        
        state = WorkflowState(
            messages=[],
            iteration=2,
            user_input="search test",
            original_user_input="search test",
            processed_output="",
            should_continue=True,
            search_results="Found results",
            search_queries=search_queries,
            parallel_search_stats={"total": 3, "successful": 2, "failed": 1},
            recent_search_mode=True,
            search_days_limit=14,
            initial_output="initial response",
            reviewed_output="reviewed response",
            document_generated=True,
            document_content="# Document\nContent here",
            document_path="/tmp/doc.md",
            slack_notification_sent=True
        )
        
        assert state["search_queries"] == search_queries
        assert len(state["search_queries"]) == 3
        assert "Python tutorial" in state["search_queries"]

    def test_workflow_state_parallel_search_stats(self):
        """Test WorkflowState with parallel search statistics."""
        stats = {
            "total_queries": 3,
            "successful_queries": 2,
            "failed_queries": 1,
            "total_time": 5.5,
            "average_time": 1.83
        }
        
        state = WorkflowState(
            messages=[],
            iteration=1,
            user_input="test",
            original_user_input="test",
            processed_output="",
            should_continue=True,
            search_results="",
            search_queries=[],
            parallel_search_stats=stats,
            recent_search_mode=False,
            search_days_limit=30,
            initial_output="",
            reviewed_output="",
            document_generated=False,
            document_content="",
            document_path="",
            slack_notification_sent=False
        )
        
        assert state["parallel_search_stats"] == stats
        assert state["parallel_search_stats"]["total_queries"] == 3
        assert state["parallel_search_stats"]["successful_queries"] == 2
        assert isinstance(state["parallel_search_stats"]["total_time"], float)

    def test_workflow_state_is_typed_dict(self):
        """Test that WorkflowState is properly typed as TypedDict."""
        # This mainly tests that the import and inheritance work correctly
        assert issubclass(WorkflowState, dict)
        
        # WorkflowState should be a TypedDict, which means it has __annotations__
        assert hasattr(WorkflowState, '__annotations__')
        assert len(WorkflowState.__annotations__) > 0
        
        # Check some key annotations exist
        annotations = WorkflowState.__annotations__
        assert 'messages' in annotations
        assert 'iteration' in annotations
        assert 'user_input' in annotations
        assert 'should_continue' in annotations
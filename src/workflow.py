"""Main workflow orchestrator for LangGraph."""

import os
from langgraph.graph import StateGraph, START, END

from .core.state import WorkflowState
from .config.langfuse_config import conditional_observe
from .nodes import (
    input_node,
    generate_search_queries,
    parallel_search_node,
    search_node,
    processing_node,
    review_node,
    documentation_node,
    slack_notification_node,
)
from .services.llm import check_ollama_connection


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow with Ollama."""
    # Create the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes to the workflow
    workflow.add_node("input", input_node)
    workflow.add_node("search", search_node)
    workflow.add_node("process", processing_node)
    workflow.add_node("review", review_node)
    workflow.add_node("document", documentation_node)
    
    # Check if Slack webhook URL is configured
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        workflow.add_node("slack_notification", slack_notification_node)

    # Define the workflow edges with conditional Slack notification
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "search")
    workflow.add_edge("search", "process")
    workflow.add_edge("process", "review")
    workflow.add_edge("review", "document")

    if slack_webhook_url:
        workflow.add_edge("document", "slack_notification")
        workflow.add_edge("slack_notification", END)
    else:
        workflow.add_edge("document", END)

    return workflow


def create_initial_state(user_question: str) -> WorkflowState:
    """Create the initial state for the workflow."""
    return {
        "messages": [],
        "iteration": 0,
        "user_input": user_question,
        "original_user_input": user_question,  # Store original question
        "processed_output": "",
        "should_continue": True,
        "search_results": "",
        "search_queries": [],  # Store generated search queries
        "parallel_search_stats": {},  # Store parallel search statistics
        "recent_search_mode": False,
        "search_days_limit": 60,  # Default to 2 months
        "initial_output": "",  # Store first AI output for comparison
        "reviewed_output": "",  # Store Claude Code reviewed output
        "document_generated": False,  # Track document generation status
        "document_content": "",  # Store generated markdown content
        "document_path": "",  # Store path to generated document
        "slack_notification_sent": not bool(
            os.getenv("SLACK_WEBHOOK_URL")
        ),  # Track Slack notification status (True if not needed)
    }


@conditional_observe(name="run_workflow")
def run_workflow(user_question: str = None) -> WorkflowState:
    """Run the complete workflow with the given user question."""
    print("🚀 Starting LangGraph Workflow with Ollama gpt-oss:20b")
    print("=" * 60)

    # Check Ollama connection
    ollama_available = check_ollama_connection()
    if not ollama_available:
        print("\n⚠️  Continuing anyway - will use fallback responses if needed")

    print()

    # Get user input if not provided
    if not user_question:
        print("💬 Please enter your question:")
        try:
            user_question = input("❓ ")
        except EOFError:
            user_question = ""

        if not user_question.strip():
            user_question = "Explain the concept of LangGraph workflows and their benefits for AI applications"
            print(f"🔄 Using default question: {user_question}")

    # Create the workflow
    workflow = create_workflow()

    # Compile the workflow
    app = workflow.compile()

    # Initial state
    initial_state = create_initial_state(user_question)

    print("\n📋 Initial State:")
    print(f"  User Input: {initial_state['user_input']}")
    print(f"  Iteration: {initial_state['iteration']}")
    print()

    # Execute the workflow
    print("⚡ Executing Workflow with Ollama...")
    print("-" * 40)

    try:
        final_state = app.invoke(initial_state)
        print("\n✅ Workflow Completed!")
        return final_state

    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise
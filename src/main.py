"""Main entry point for the LangGraph workflow application."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from .workflow import run_workflow

# Load environment variables
load_dotenv()


def display_workflow_results(final_state):
    """Display the results of the workflow execution."""
    print("=" * 60)
    print("📊 Final Results:")
    print(f"  Total Iterations: {final_state['iteration']}")
    print(f"  Message Count: {len(final_state['messages'])}")
    print(
        f"  Document Generated: {'✅' if final_state.get('document_generated', False) else '❌'}"
    )
    print(
        f"  Slack Notification: {'✅' if final_state.get('slack_notification_sent', False) else '❌'}"
    )
    print()

    print("💬 Full Conversation History:")
    for i, message in enumerate(final_state["messages"], 1):
        message_type = (
            "User" if isinstance(message, HumanMessage) else "AI (gpt-oss:20b)"
        )
        content = message.content
        print(f"  {i}. [{message_type}]:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        print()

    # Display review results if available
    if final_state.get("reviewed_output"):
        print("🔍 Claude Code Review Results:")
        print("=" * 60)
        print(final_state["reviewed_output"])
        print("=" * 60)
        print()

    # Display documentation status
    if final_state.get("document_generated"):
        print("📝 Documentation successfully generated in Docs/ directory")
        print(f"📄 Document path: {final_state.get('document_path', 'Unknown')}")
    else:
        print("⚠️ Documentation generation failed or was skipped")

    # Display Slack notification status
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        if final_state.get("slack_notification_sent"):
            print(
                "📢 Slack notification sent successfully with complete document content"
            )
        else:
            print(
                "⚠️ Slack notification failed (check SLACK_WEBHOOK_URL environment variable)"
            )
    else:
        print("ℹ️ Slack notification skipped (SLACK_WEBHOOK_URL not configured)")


def main():
    """Main function to run the workflow with Ollama."""
    try:
        final_state = run_workflow()
        display_workflow_results(final_state)
        return 0
    except Exception as e:
        print(f"❌ Application failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
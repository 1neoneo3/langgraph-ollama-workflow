#!/usr/bin/env python3
"""Test script to see full Ollama responses."""

import sys

sys.path.insert(0, ".")

from ollama_workflow import create_workflow


def main():
    """Test the workflow and show full responses."""
    print("🚀 Testing Full LLM Responses with Ollama gpt-oss:20b")
    print("=" * 60)

    workflow = create_workflow()
    app = workflow.compile()

    initial_state = {
        "messages": [],
        "iteration": 0,
        "user_input": "What are the key advantages of using LangGraph for complex AI workflows?",
        "processed_output": "",
        "should_continue": True,
    }

    try:
        final_state = app.invoke(initial_state)

        print("\n💬 Full Conversation History:")
        print("=" * 60)

        for i, message in enumerate(final_state["messages"], 1):
            message_type = (
                "👤 User"
                if hasattr(message, "content")
                and message.__class__.__name__ == "HumanMessage"
                else "🤖 AI (gpt-oss:20b)"
            )
            print(f"\n{i}. {message_type}:")
            print("-" * 40)
            print(message.content)
            print("-" * 40)

        print("\n📊 Summary:")
        print(f"  Total Iterations: {final_state['iteration']}")
        print(f"  Messages: {len(final_state['messages'])}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

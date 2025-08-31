#!/usr/bin/env python3
"""Custom test with a different question."""

import sys

sys.path.insert(0, ".")

from ollama_workflow import create_workflow


def main():
    """Test the workflow with a custom technical question."""
    print("🧪 Custom LangGraph Workflow Test")
    print("=" * 50)

    workflow = create_workflow()
    app = workflow.compile()

    initial_state = {
        "messages": [],
        "iteration": 0,
        "user_input": "How would you implement a multi-agent system using LangGraph where different AI agents collaborate to solve complex problems?",
        "processed_output": "",
        "should_continue": True,
    }

    try:
        final_state = app.invoke(initial_state)

        print("\n🤖 AI Agent Response Summary:")
        print("=" * 50)

        for i, message in enumerate(final_state["messages"], 1):
            if (
                hasattr(message, "content")
                and message.__class__.__name__ == "AIMessage"
            ):
                content = message.content
                # Show first 300 characters to get a good preview
                preview = content[:300] + ("..." if len(content) > 300 else "")
                print(f"\n📝 Response {i // 2}:")
                print(f"   {preview}")

        print("\n📈 Workflow Metrics:")
        print(f"   Iterations: {final_state['iteration']}")
        print(f"   Total Messages: {len(final_state['messages'])}")
        print(
            f"   User Questions: {sum(1 for m in final_state['messages'] if m.__class__.__name__ == 'HumanMessage')}"
        )
        print(
            f"   AI Responses: {sum(1 for m in final_state['messages'] if m.__class__.__name__ == 'AIMessage')}"
        )

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

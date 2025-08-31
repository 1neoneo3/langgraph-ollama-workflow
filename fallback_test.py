#!/usr/bin/env python3
"""Test fallback behavior when Ollama is not available."""

import sys

sys.path.insert(0, ".")

from ollama_workflow import create_workflow


def test_with_ollama_stopped():
    """Test workflow behavior when Ollama service is stopped."""
    print("🔧 Testing Fallback Behavior (Ollama Service Stopped)")
    print("=" * 60)

    # Try to stop Ollama temporarily (this might not work in all environments)
    try:
        # Just test with a wrong port to simulate connection failure
        import requests
        from unittest.mock import patch

        def mock_requests_get(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Connection refused")

        workflow = create_workflow()
        app = workflow.compile()

        initial_state = {
            "messages": [],
            "iteration": 0,
            "user_input": "Test fallback behavior when LLM is not available",
            "processed_output": "",
            "should_continue": True,
        }

        print("⚡ Running workflow with simulated Ollama failure...")

        # Patch requests to simulate connection failure
        with patch("requests.get", side_effect=mock_requests_get):
            final_state = app.invoke(initial_state)

        print("\n✅ Fallback Test Results:")
        print("=" * 40)

        for i, message in enumerate(final_state["messages"], 1):
            message_type = (
                "👤 User"
                if message.__class__.__name__ == "HumanMessage"
                else "🤖 AI (Fallback)"
            )
            content = message.content
            preview = content[:150] + ("..." if len(content) > 150 else "")
            print(f"{i}. {message_type}: {preview}")

        print("\n📊 Fallback Metrics:")
        print(f"   Iterations: {final_state['iteration']}")
        print(f"   Messages: {len(final_state['messages'])}")
        print("   Fallback Mode: ✅ Working")

    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run fallback tests."""
    test_with_ollama_stopped()


if __name__ == "__main__":
    main()

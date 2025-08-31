#!/usr/bin/env python3
"""Performance test for the Ollama LangGraph workflow."""

import sys
import time

sys.path.insert(0, ".")

from ollama_workflow import create_workflow


def main():
    """Run performance test with timing measurements."""
    print("⚡ LangGraph + Ollama Performance Test")
    print("=" * 50)

    workflow = create_workflow()
    app = workflow.compile()

    initial_state = {
        "messages": [],
        "iteration": 0,
        "user_input": "Explain the performance characteristics of large language models in production environments.",
        "processed_output": "",
        "should_continue": True,
    }

    print("🚀 Starting timed execution...")
    start_time = time.time()

    try:
        final_state = app.invoke(initial_state)

        end_time = time.time()
        execution_time = end_time - start_time

        print("\n📊 Performance Results:")
        print("=" * 30)
        print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")
        print(f"🔄 Iterations Completed: {final_state['iteration']}")
        print(f"💬 Messages Generated: {len(final_state['messages'])}")
        print(
            f"⚡ Average Time per Iteration: {execution_time / final_state['iteration']:.2f}s"
        )

        # Calculate response lengths
        ai_messages = [
            m for m in final_state["messages"] if m.__class__.__name__ == "AIMessage"
        ]
        total_chars = sum(len(m.content) for m in ai_messages)
        avg_response_length = total_chars / len(ai_messages) if ai_messages else 0

        print(f"📝 Average Response Length: {avg_response_length:.0f} characters")
        print(f"🎯 Characters per Second: {total_chars / execution_time:.1f}")

        # Show sample of the first response
        if ai_messages:
            first_response = ai_messages[0].content
            sample = first_response[:200] + ("..." if len(first_response) > 200 else "")
            print("\n📄 Sample Response:")
            print(f"   {sample}")

        # Performance rating
        if execution_time < 10:
            rating = "🚀 Excellent"
        elif execution_time < 30:
            rating = "✅ Good"
        elif execution_time < 60:
            rating = "⚠️  Acceptable"
        else:
            rating = "🐌 Needs Optimization"

        print(f"\n🏆 Performance Rating: {rating}")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LangGraph workflow implementation with Ollama gpt-oss:20b model."""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()


# Define the state structure for our workflow
class WorkflowState(TypedDict):
    messages: list[BaseMessage]
    iteration: int
    user_input: str
    processed_output: str
    should_continue: bool


def input_node(state: WorkflowState) -> WorkflowState:
    """Process initial user input."""
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    # Add user message to conversation
    if user_input:
        messages.append(HumanMessage(content=user_input))

    return {
        **state,
        "messages": messages,
        "iteration": state.get("iteration", 0) + 1,
    }


def processing_node(state: WorkflowState) -> WorkflowState:
    """Process the user input using Ollama gpt-oss:20b model."""
    messages = state["messages"]
    iteration = state["iteration"]

    if not messages:
        return state

    print(f"🤖 Processing iteration {iteration} with Ollama gpt-oss:20b...")

    try:
        # Initialize Ollama with gpt-oss:20b model
        llm = ChatOllama(
            model="gpt-oss:20b",
            base_url="http://localhost:11434",  # Default Ollama port
            temperature=0.7,
        )

        # Create a focused prompt for the LLM
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            content = last_message.content

            # Create a system prompt for better responses
            system_prompt = f"""
            You are an AI assistant processing iteration {iteration} of a LangGraph workflow.
            Please provide a thoughtful response to the user's input.
            Keep your response concise but informative.
            
            User input: {content}
            """

            # Get response from Ollama
            response = llm.invoke([HumanMessage(content=system_prompt)])

            # Add AI response to messages
            ai_response = response.content
            messages.append(AIMessage(content=ai_response))

            print("✅ LLM Full Response:")
            print("-" * 60)
            print(ai_response)
            print("-" * 60)

            return {
                **state,
                "messages": messages,
                "processed_output": ai_response,
            }

    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        print("🔄 Falling back to simple response generation...")

        # Fallback to simple processing if Ollama is not available
        if messages and isinstance(messages[-1], HumanMessage):
            content = messages[-1].content
            fallback_response = (
                f"Processing iteration {iteration}: {content} (Ollama unavailable)"
            )
            messages.append(AIMessage(content=fallback_response))

            return {
                **state,
                "messages": messages,
                "processed_output": fallback_response,
            }

    return state


def decision_node(state: WorkflowState) -> WorkflowState:
    """Decide whether to continue processing or end."""
    iteration = state.get("iteration", 0)

    # Continue for fewer iterations when using LLM (to avoid long execution)
    should_continue = iteration < 2

    return {
        **state,
        "should_continue": should_continue,
    }


def continuation_node(state: WorkflowState) -> WorkflowState:
    """Continue processing if needed."""
    messages = state["messages"]
    iteration = state["iteration"]

    # Add a continuation message that gives the LLM more context
    continuation_message = f"Please elaborate or provide additional insights based on your previous response. This is continuation {iteration}."

    return {
        **state,
        "messages": messages,
        "user_input": continuation_message,
    }


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow with Ollama."""

    # Create the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes to the workflow
    workflow.add_node("input", input_node)
    workflow.add_node("process", processing_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("continue", continuation_node)

    # Define the workflow edges
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "process")
    workflow.add_edge("process", "decision")

    # Conditional routing function
    def route_decision(state: WorkflowState) -> str:
        """Route based on the decision state."""
        return "continue" if state.get("should_continue", False) else "end"

    # Conditional edges from decision node
    workflow.add_conditional_edges(
        "decision",
        route_decision,
        {
            "continue": "continue",
            "end": END,
        },
    )

    # Edge from continue back to input for loop
    workflow.add_edge("continue", "input")

    return workflow


def check_ollama_connection():
    """Check if Ollama is running and gpt-oss:20b is available."""
    try:
        import requests

        print("🔍 Checking Ollama connection...")

        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model["name"] for model in models.get("models", [])]

            print(f"✅ Ollama is running with {len(model_names)} models")

            if "gpt-oss:20b" in model_names:
                print("✅ gpt-oss:20b model is available")
                return True
            else:
                print("❌ gpt-oss:20b model not found")
                print("Available models:", model_names)
                print("\n💡 To install gpt-oss:20b, run: ollama pull gpt-oss:20b")
                return False
        else:
            print("❌ Ollama API returned error:", response.status_code)
            return False

    except requests.exceptions.RequestException as e:
        print("❌ Cannot connect to Ollama:", e)
        print("\n💡 Make sure Ollama is running: ollama serve")
        return False
    except ImportError:
        print("❌ requests library not available for Ollama check")
        return False


def main():
    """Main function to run the workflow with Ollama."""
    print("🚀 Starting LangGraph Workflow with Ollama gpt-oss:20b")
    print("=" * 60)

    # Check Ollama connection
    ollama_available = check_ollama_connection()
    if not ollama_available:
        print("\n⚠️  Continuing anyway - will use fallback responses if needed")

    print()

    # Get user input
    print("💬 Please enter your question:")
    user_question = input("❓ ")
    
    if not user_question.strip():
        user_question = "Explain the concept of LangGraph workflows and their benefits for AI applications"
        print(f"🔄 Using default question: {user_question}")

    # Create the workflow
    workflow = create_workflow()

    # Compile the workflow
    app = workflow.compile()

    # Initial state
    initial_state = {
        "messages": [],
        "iteration": 0,
        "user_input": user_question,
        "processed_output": "",
        "should_continue": True,
    }

    print(f"\n📋 Initial State:")
    print(f"  User Input: {initial_state['user_input']}")
    print(f"  Iteration: {initial_state['iteration']}")
    print()

    # Execute the workflow
    print("⚡ Executing Workflow with Ollama...")
    print("-" * 40)

    try:
        final_state = app.invoke(initial_state)

        print("\n✅ Workflow Completed!")
        print("=" * 60)
        print("📊 Final Results:")
        print(f"  Total Iterations: {final_state['iteration']}")
        print(f"  Message Count: {len(final_state['messages'])}")
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

    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""Simple LangGraph workflow implementation based on search results."""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
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
    """Process the user input and generate a response."""
    messages = state["messages"]
    iteration = state["iteration"]

    # Simple processing logic - in real implementation, this would use LLM
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            content = last_message.content

            # Simple response generation
            response = f"Processing iteration {iteration}: {content}"

            # Add AI response to messages
            messages.append(AIMessage(content=response))

            return {
                **state,
                "messages": messages,
                "processed_output": response,
            }

    return state


def decision_node(state: WorkflowState) -> WorkflowState:
    """Decide whether to continue processing or end."""
    iteration = state.get("iteration", 0)

    # Add decision info to state
    should_continue = iteration < 3

    return {
        **state,
        "should_continue": should_continue,
    }


def continuation_node(state: WorkflowState) -> WorkflowState:
    """Continue processing if needed."""
    messages = state["messages"]

    # Add a continuation message
    continuation_message = f"Continuing workflow... (iteration {state['iteration']})"
    messages.append(AIMessage(content=continuation_message))

    return {
        **state,
        "messages": messages,
        "user_input": f"Continue processing from iteration {state['iteration']}",
    }


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow."""

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


def main():
    """Main function to run the workflow."""
    print("🚀 Starting LangGraph Workflow Implementation")
    print("=" * 50)

    # Create the workflow
    workflow = create_workflow()

    # Compile the workflow
    app = workflow.compile()

    # Initial state
    initial_state = {
        "messages": [],
        "iteration": 0,
        "user_input": "Hello, please process this LangGraph workflow example",
        "processed_output": "",
        "should_continue": True,
    }

    print("📋 Initial State:")
    print(f"  User Input: {initial_state['user_input']}")
    print(f"  Iteration: {initial_state['iteration']}")
    print()

    # Execute the workflow
    print("⚡ Executing Workflow...")
    print("-" * 30)

    try:
        final_state = app.invoke(initial_state)

        print("✅ Workflow Completed!")
        print("=" * 50)
        print("📊 Final Results:")
        print(f"  Total Iterations: {final_state['iteration']}")
        print(f"  Final Output: {final_state['processed_output']}")
        print(f"  Message Count: {len(final_state['messages'])}")
        print()

        print("💬 Conversation History:")
        for i, message in enumerate(final_state["messages"], 1):
            message_type = "User" if isinstance(message, HumanMessage) else "AI"
            print(f"  {i}. [{message_type}]: {message.content}")

    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

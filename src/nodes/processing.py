"""Processing node for handling LLM interactions."""

from langchain_core.messages import HumanMessage, AIMessage

from ..config.settings import Config
from ..config.langfuse_config import conditional_observe
from ..core.state import WorkflowState
from ..services.llm import create_ollama_llm, handle_ollama_fallback
from ..utils.datetime_utils import get_current_datetime_info
from ..utils.helpers import create_system_prompt


@conditional_observe(name="processing_node")
def processing_node(state: WorkflowState) -> WorkflowState:
    """Process the user input using Ollama gpt-oss:20b model with search results."""
    messages = state["messages"]
    iteration = state["iteration"]
    search_results = state.get("search_results", "")

    if not messages:
        return state

    print(f"🤖 Processing iteration {iteration} with Ollama {Config.OLLAMA_MODEL}...")

    try:
        llm = create_ollama_llm()
        last_message = messages[-1]

        if isinstance(last_message, HumanMessage):
            content = last_message.content
            current_date_info = get_current_datetime_info()

            # Create system prompt using helper function
            system_prompt = create_system_prompt(
                content, current_date_info, search_results, iteration
            )

            # Get response from Ollama
            response = llm.invoke([HumanMessage(content=system_prompt)])
            ai_response = (
                response.content if response.content else "応答を生成できませんでした。"
            )
            messages.append(AIMessage(content=ai_response))

            print("✅ LLM Full Response:")
            print("-" * 60)
            print(ai_response)
            print("-" * 60)

            return {
                **state,
                "messages": messages,
                "processed_output": ai_response,
                "initial_output": ai_response,
            }

    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        print("🔄 Falling back to simple response generation...")

        fallback_result = handle_ollama_fallback(messages, iteration)
        return {**state, **fallback_result}

    return state
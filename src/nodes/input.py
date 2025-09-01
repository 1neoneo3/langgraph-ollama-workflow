"""Input node for processing user input and detecting search mode."""

from langchain_core.messages import HumanMessage

from ..core.state import WorkflowState
from ..utils.datetime_utils import get_current_datetime_info, detect_recent_search_mode


def input_node(state: WorkflowState) -> WorkflowState:
    """Process initial user input and detect recent search keywords."""
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    # Add user message to conversation
    if user_input:
        messages.append(HumanMessage(content=user_input))

    current_date_info = get_current_datetime_info()
    recent_search_mode, search_days_limit = detect_recent_search_mode(
        user_input, current_date_info
    )

    return {
        **state,
        "messages": messages,
        "iteration": state.get("iteration", 0) + 1,
        "recent_search_mode": recent_search_mode,
        "search_days_limit": search_days_limit,
        "original_user_input": state.get("original_user_input", user_input),
    }
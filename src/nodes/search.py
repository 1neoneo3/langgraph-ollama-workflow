"""Search node for handling single search operations."""

from ..config.settings import Config
from ..config.langfuse_config import conditional_observe
from ..core.state import WorkflowState
from ..services.search import perform_search


@conditional_observe(name="search_node")
def search_node(state: WorkflowState) -> WorkflowState:
    """Perform a single search operation based on user input."""
    user_input = state.get("user_input", "")
    recent_search_mode = state.get("recent_search_mode", False)
    search_days_limit = state.get("search_days_limit", 60)
    
    print(f"🔍 Performing search for: {user_input[:100]}...")
    
    try:
        # Perform single search operation
        search_results = perform_search(
            query=user_input,
            recent_search_mode=recent_search_mode,
            days_limit=search_days_limit
        )
        
        print(f"✅ Search completed. Results length: {len(search_results)} characters")
        
        return {
            **state,
            "search_results": search_results,
        }
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        fallback_message = f"検索に失敗しましたが、質問「{user_input}」について利用可能な知識で回答いたします。"
        
        return {
            **state,
            "search_results": fallback_message,
        }
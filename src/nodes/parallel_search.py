"""Parallel search node for executing multiple searches concurrently."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config.settings import Config
from ..core.state import WorkflowState
from ..services.search import execute_single_search
from ..services.review import execute_websearch_fallback
from ..utils.helpers import format_parallel_search_results


def parallel_search_node(state: WorkflowState) -> WorkflowState:
    """Execute multiple searches in parallel using the generated queries."""
    search_queries = state.get("search_queries", [])
    recent_search_mode = state.get("recent_search_mode", False)
    search_days_limit = state.get("search_days_limit", Config.DEFAULT_SEARCH_DAYS_LIMIT)

    if not search_queries:
        print("⚠️ No search queries available")
        return {**state, "search_results": ""}

    print(f"🔍 Executing {len(search_queries)} parallel searches...")

    search_results = []
    total_start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # Submit all search tasks
            future_to_query = {
                executor.submit(
                    execute_single_search,
                    (i, query),
                    recent_search_mode,
                    search_days_limit,
                ): (i, query)
                for i, query in enumerate(search_queries)
            }

            # Collect results as they complete
            for future in as_completed(future_to_query):
                query_index, query = future_to_query[future]
                try:
                    result = future.result()
                    search_results.append(result)
                except Exception as exc:
                    print(f"❌ Search {query_index + 1} generated exception: {exc}")
                    search_results.append(
                        {
                            "query": query,
                            "results": f"Exception: {str(exc)}",
                            "success": False,
                            "elapsed_time": 0,
                        }
                    )

    except Exception as e:
        print(f"❌ Parallel search execution error: {e}")
        return {**state, "search_results": f"Parallel search error: {str(e)}"}

    total_elapsed_time = time.time() - total_start_time
    successful_searches = [r for r in search_results if r["success"]]
    failed_searches = [r for r in search_results if not r["success"]]

    print("📊 Search Summary:")
    print(f"  ✅ Successful: {len(successful_searches)}/{len(search_queries)}")
    print(f"  ❌ Failed: {len(failed_searches)}")
    print(f"  ⏱️ Total time: {total_elapsed_time:.2f}s")

    # If all searches failed, use WebSearch as fallback
    if len(successful_searches) == 0:
        print("🔄 All parallel searches failed - falling back to Claude Code WebSearch")

        try:
            websearch_results = execute_websearch_fallback(search_queries)

            print("✅ WebSearch fallback completed")
            print(f"📄 WebSearch results length: {len(websearch_results)} characters")

            # Create fallback results
            main_query = (
                search_queries[0] if search_queries else state.get("user_input", "")
            )
            combined_results = (
                "WebSearch Fallback Results (all parallel searches failed):\n\n"
            )
            combined_results += f"🌐 WebSearch Query: {main_query}\n"
            combined_results += (
                f"⏱️ Fallback execution time: {total_elapsed_time:.2f}s\n"
            )
            combined_results += f"📊 Results:\n{websearch_results}\n"
            combined_results += "-" * 50 + "\n\n"
            combined_results += "Original parallel search failures:\n"

            for i, result in enumerate(search_results, 1):
                combined_results += (
                    f"❌ Search {i}: {result['query']} - {result['results']}\n"
                )

            return {
                **state,
                "search_results": combined_results,
                "parallel_search_stats": {
                    "total_queries": len(search_queries),
                    "successful": 0,
                    "failed": len(failed_searches),
                    "total_time": total_elapsed_time,
                    "websearch_fallback": True,
                },
            }

        except ImportError:
            print("❌ Claude Code SDK not available for WebSearch fallback")
        except Exception as e:
            print(f"❌ WebSearch fallback failed: {e}")

    # Format results using helper function
    combined_results = format_parallel_search_results(
        search_results, total_elapsed_time
    )

    return {
        **state,
        "search_results": combined_results,
        "parallel_search_stats": {
            "total_queries": len(search_queries),
            "successful": len(successful_searches),
            "failed": len(failed_searches),
            "total_time": total_elapsed_time,
        },
    }
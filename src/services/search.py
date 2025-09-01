"""Search service for psearch and parallel search functionality."""

import subprocess
import sys
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config.settings import Config
from ..utils.datetime_utils import get_time_description
from ..utils.helpers import build_psearch_command


def execute_psearch_with_progress(psearch_cmd: List[str]) -> Dict[str, any]:
    """Execute psearch command with real-time progress display."""
    start_time = time.time()

    try:
        process = subprocess.Popen(
            psearch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_lines = []

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(f"📤 {output.rstrip()}")
                sys.stdout.flush()
                stdout_lines.append(output)

        stderr_output = process.stderr.read()
        return_code = process.wait()
        elapsed_time = time.time() - start_time

        return {
            "success": return_code == 0,
            "stdout": "".join(stdout_lines),
            "stderr": stderr_output,
            "elapsed_time": elapsed_time,
            "return_code": return_code,
        }

    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "elapsed_time": time.time() - start_time,
            "return_code": -1,
            "error": e,
        }


def execute_single_search(
    query_info: tuple, recent_search_mode: bool, search_days_limit: int
) -> Dict[str, any]:
    """Execute a single search with proper error handling."""
    query_index, query = query_info
    print(f"🔎 Search {query_index + 1}: {query}")

    try:
        # Build psearch command using existing helper
        psearch_cmd = build_psearch_command(query, recent_search_mode, search_days_limit)
        # Override some settings for parallel search
        psearch_cmd[4] = "3"  # Change -n to 3 for parallel searches

        # Execute search with timeout
        start_time = time.time()
        result = subprocess.run(
            psearch_cmd, capture_output=True, text=True, timeout=Config.SEARCH_TIMEOUT
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Search {query_index + 1} completed in {elapsed_time:.2f}s")
            return {
                "query": query,
                "results": result.stdout,
                "success": True,
                "elapsed_time": elapsed_time,
            }
        else:
            print(f"❌ Search {query_index + 1} failed: {result.stderr}")
            return {
                "query": query,
                "results": f"Search failed: {result.stderr}",
                "success": False,
                "elapsed_time": elapsed_time,
            }

    except subprocess.TimeoutExpired:
        print(f"⏰ Search {query_index + 1} timed out")
        return {
            "query": query,
            "results": "Search timed out",
            "success": False,
            "elapsed_time": Config.SEARCH_TIMEOUT,
        }
    except Exception as e:
        print(f"❌ Search {query_index + 1} error: {e}")
        return {
            "query": query,
            "results": f"Search error: {str(e)}",
            "success": False,
            "elapsed_time": 0,
        }


def execute_parallel_searches(
    search_queries: List[str], recent_search_mode: bool, search_days_limit: int
) -> tuple[List[Dict[str, any]], float]:
    """Execute multiple searches in parallel."""
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
        raise

    total_elapsed_time = time.time() - total_start_time
    return search_results, total_elapsed_time


def generate_search_queries_fallback(user_input: str) -> List[str]:
    """Generate fallback search queries when Claude Code SDK is not available."""
    return [
        user_input,  # Basic query
        f"{user_input} 最新",  # Latest info query
        f"{user_input} 実装方法",  # Implementation query
    ]


def perform_search(query: str, recent_search_mode: bool = False, days_limit: int = 60) -> str:
    """Perform a single search operation using psearch."""
    try:
        # Build psearch command using existing helper
        psearch_cmd = build_psearch_command(query, recent_search_mode, days_limit)
        
        # Execute search with progress
        result = execute_psearch_with_progress(psearch_cmd)
        
        if result["success"]:
            return result["stdout"]
        else:
            return f"Search failed: {result['stderr']}"
            
    except Exception as e:
        return f"Search error: {str(e)}"
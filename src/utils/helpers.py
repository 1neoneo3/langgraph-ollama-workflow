"""General helper functions."""

from typing import Dict, List
from ..config.settings import Config


def create_system_prompt(
    content: str, current_date_info: Dict[str, any], search_results: str, iteration: int
) -> str:
    """Create standardized system prompt for LLM calls."""
    return f"""
【重要な指示】
- すべての回答は日本語で記述してください
- 現在日時: {current_date_info["date_str"]} ({current_date_info["year"]}年)
- 最新情報（{current_date_info["year"] - 1}年以降）を優先して活用してください

あなたはLangGraphワークフローの{iteration}回目の処理を行うAIアシスタントです。
最新の検索結果にアクセスして、正確で最新の情報を提供することができます。
ユーザーの入力に対して、必要に応じて検索結果から関連情報を取り入れた、思慮深い回答を日本語で提供してください。
簡潔でありながら、情報量豊富な回答を心がけてください。

【現在の日時情報】
現在は{current_date_info["date_str"]}（{current_date_info["year"]}年）です。この日時を考慮して、最新の情報を優先して回答してください。

ユーザーの入力: {content}

検索結果 (利用可能な場合):
{search_results if search_results else "検索結果がありません"}

【回答要件】
- すべて日本語で回答してください
- 検索結果を活用して、{current_date_info["year"]}年時点での最新で正確な情報を含めてください
- 古い情報（{current_date_info["year"] - 2}年以前）がある場合は、最新動向も併記してください
- 技術的な内容の場合は、最新バージョンや仕様変更も考慮してください
"""


def build_psearch_command(
    query: str, recent_search_mode: bool, search_days_limit: int
) -> List[str]:
    """Build psearch command with appropriate filters."""
    psearch_cmd = ["psearch", "search", query[:100], "-n", "5", "-c", "--json"]

    if recent_search_mode:
        if search_days_limit <= 30:
            psearch_cmd.extend(["-r", "-s"])
        else:
            months = max(1, search_days_limit // 30)
            psearch_cmd.extend(["-r", "--months", str(months), "-s"])

    return psearch_cmd


def format_parallel_search_results(
    search_results: List[Dict[str, any]], total_elapsed_time: float
) -> str:
    """Format parallel search results into a readable summary."""
    successful_searches = [r for r in search_results if r["success"]]

    combined_results = f"Parallel Search Results ({len(successful_searches)}/{len(search_results)} successful):\n\n"

    for i, result in enumerate(search_results, 1):
        status = "✅" if result["success"] else "❌"
        combined_results += f"{status} Search {i}: {result['query']}\n"
        combined_results += f"Time: {result['elapsed_time']:.2f}s\n"

        if result["success"] and result["results"]:
            limited_results = result["results"][: Config.INDIVIDUAL_RESULT_LIMIT]
            if len(result["results"]) > Config.INDIVIDUAL_RESULT_LIMIT:
                limited_results += "..."
            combined_results += f"Results:\n{limited_results}\n"
        else:
            combined_results += f"Error: {result['results']}\n"
        combined_results += "-" * 50 + "\n\n"

    return combined_results
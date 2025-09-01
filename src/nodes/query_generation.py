"""Query generation node for creating diverse search queries."""

import asyncio
import re

from ..core.state import WorkflowState


def generate_search_queries_fallback(user_input: str) -> list[str]:
    """Generate fallback search queries when Claude Code SDK is not available."""
    return [
        user_input,  # Basic query
        f"{user_input} 最新",  # Latest info query
        f"{user_input} 実装方法",  # Implementation query
    ]


def generate_search_queries(state: WorkflowState) -> WorkflowState:
    """Generate exactly 3 diverse search queries using Claude Code agent."""
    user_input = state.get("user_input", "")

    if not user_input:
        return {**state, "search_queries": []}

    print(f"🧠 Generating 3 search queries using Claude Code agent for: {user_input}")

    try:
        # Use Claude Code SDK to generate diverse search queries
        print("📦 Using Claude Code agent for query generation...")
        from claude_code_sdk import query as claude_query, ClaudeCodeOptions

        # Configure options for Claude Code
        query_generation_prompt = f"""あなたは検索戦略の専門家です。

以下の質問に対して、効果的な検索を行うために3つの異なる角度からの検索クエリを生成してください。

元の質問: {user_input}

以下の要件に従って、ちょうど3つの検索クエリを生成してください：

1. **基本概念クエリ**: 核となる概念や定義を探すクエリ
2. **最新情報クエリ**: 最新の動向や更新情報を探すクエリ  
3. **実践的クエリ**: 実装例や具体的な使用例を探すクエリ

各クエリは50文字以内で、簡潔かつ効果的にしてください。

以下の形式で出力してください：
クエリ1: [検索クエリ1]
クエリ2: [検索クエリ2]
クエリ3: [検索クエリ3]"""

        options = ClaudeCodeOptions(
            system_prompt="あなたは検索戦略の専門家です。与えられた質問に対して、3つの異なる角度から効果的な検索クエリを生成してください。",
            max_turns=1,
        )

        async def get_queries():
            content = ""
            async for message in claude_query(
                prompt=query_generation_prompt, options=options
            ):
                if hasattr(message, "content"):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if hasattr(block, "text"):
                                content += block.text
                    else:
                        content += str(message.content)
            return content

        # Run async function
        query_response = asyncio.run(get_queries())

        # Extract queries from the response
        query_pattern = r"クエリ\d+:\s*(.+)"
        matches = re.findall(query_pattern, query_response)

        # Clean up and limit to exactly 3 queries
        queries = [q.strip() for q in matches if q.strip()]
        queries = queries[:3]  # Limit to exactly 3 queries

        # If we got less than 3 queries, create fallback queries
        if len(queries) < 3:
            print("⚠️ Claude Code agent returned fewer than 3 queries, using fallback")
            queries = generate_search_queries_fallback(user_input)

        # Ensure we have exactly 3 queries
        queries = queries[:3]

        print(
            f"✅ Generated exactly {len(queries)} search queries using Claude Code agent:"
        )
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")

        return {**state, "search_queries": queries}

    except ImportError:
        print("❌ Claude Code SDK not available, falling back to rule-based generation")
        fallback_queries = generate_search_queries_fallback(user_input)
        return {**state, "search_queries": fallback_queries}

    except Exception as e:
        print(f"❌ Error with Claude Code agent: {e}")
        fallback_queries = generate_search_queries_fallback(user_input)
        return {**state, "search_queries": fallback_queries}
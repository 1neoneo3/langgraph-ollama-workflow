"""Review service for Claude Code SDK integration."""

import asyncio
from typing import Dict, List

from ..config.settings import Config
from ..core.state import WorkflowState


def create_claude_code_options(
    system_prompt: str, max_turns: int = None, allowed_tools: List[str] = None
):
    """Create standardized Claude Code options."""
    from claude_code_sdk import ClaudeCodeOptions

    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        max_turns=max_turns or Config.CLAUDE_MAX_TURNS,
        allowed_tools=allowed_tools or ["WebSearch"],
    )

    # Add context7 MCP server if needed
    if "context7" in system_prompt.lower() or not allowed_tools:
        options.mcp_servers = {
            "context7": {"command": "npx", "args": ["-y", "@context7/server"]}
        }

    return options


def create_review_system_prompt(
    processed_output: str, original_question: str, current_date_info: Dict[str, any]
) -> str:
    """Create detailed system prompt for Claude Code review."""
    return f"""あなたは技術文書の校正・レビューの専門家です。

上記の回答内容について、詳細なレビューを行い、その後にレビューを反映した修正版を作成してください。

【レビュー対象の回答内容】
{processed_output}

【元の質問】
{original_question}

【重要な指示】
- 必ず日本語でレビューと修正版を作成してください
- すべての出力は日本語で記述してください 
- WebSearchツールを使用する場合も、結果は日本語でまとめてください
- 現在日時: {current_date_info["date_str"]} ({current_date_info["year"]}年)
- 最新情報（{current_date_info["year"] - 1}年以降）を優先して参照してください

【個別の指示】
- レビュー対象の文章をもとに、必ず「詳細なレビュー」と「修正版」の両方を作成してください
- レビューで指摘した内容は修正版にすべて反映してください
- すべての出力とドキュメント内容は日本語で記述してください

---

## レビューポイント

### 1. 一般的なポイント
1. 事実の正確性（間違いや古い情報がないか、特に{current_date_info["year"] - 1}年以降の最新情報との整合性）
2. 論理的な一貫性（矛盾がないか）
3. 完全性（重要な情報が抜けていないか）
4. わかりやすさ（説明が明確か）
5. 最新性（{current_date_info["year"]}年の最新情報に基づいているか）

### 2. 技術的質問の場合の追加ポイント
- 技術的正確性（コード構文、APIの使用方法、{current_date_info["year"]}年時点での最新仕様）
- ベストプラクティス準拠（業界標準に従っているか）
- セキュリティ（リスクや問題がないか）
- パフォーマンス（効率的で最適化されているか）
- 公式ドキュメントとの整合性（{current_date_info["year"]}年の最新ドキュメントに基づくか）
- 実装上の注意点や落とし穴（最新バージョンの変更点を含む）

### 3. 最新情報確認
- WebSearchツールを使用して{current_date_info["year"]}年の最新情報を確認すること
- 古い情報（{current_date_info["year"] - 2}年以前）は最新情報で補完すること
- バージョンアップやAPI変更など最新動向を反映すること
- WebSearchの結果は必ず日本語でまとめること

---

## 出力形式（必ず日本語で記述）

1. **詳細なレビュー**（箇条書き・具体的に、日本語で省略せず）
2. **修正版文章**（レビュー内容を完全に反映した修正版、日本語で完全に書く）
3. **修正点の説明**（レビュー内容に沿って何をどう修正したか、日本語で詳細に）
4. 問題がなければ「レビュー完了：問題なし（{current_date_info["date_str"]}時点）」と記述

---

【最重要】
- すべての出力は日本語で記述してください
- WebSearchの結果や引用も日本語でまとめてください
- 英語のテキストは含めないでください
- レビューと修正版は必ずセットで日本語で生成してください
- 技術的な情報は省略せず、最新情報（{current_date_info["year"]}年）に基づく更新点を明示してください"""


async def execute_claude_code_query(prompt: str, options) -> str:
    """Execute Claude Code query and return content."""
    from claude_code_sdk import query

    content = ""
    message_count = 0

    try:
        async for message in query(prompt=prompt, options=options):
            message_count += 1
            print(f"📨 Received message #{message_count} from Claude Code SDK")

            if hasattr(message, "content"):
                if isinstance(message.content, list):
                    for i, block in enumerate(message.content):
                        print(
                            f"📄 Processing content block #{i + 1} - Type: {type(block).__name__}"
                        )

                        try:
                            from claude_code_sdk.types import (
                                TextBlock,
                                ToolUseBlock,
                                ToolResultBlock,
                            )

                            if isinstance(block, TextBlock):
                                content += block.text
                            elif isinstance(block, ToolUseBlock):
                                tool_name = getattr(block, "name", "unknown")
                                print(f"🔧 ToolUseBlock - Tool: {tool_name}")
                                content += f"\n[ツール使用: {tool_name}]\n"
                            elif isinstance(block, ToolResultBlock):
                                tool_result = str(
                                    getattr(block, "content", "no result")
                                )
                                print(
                                    f"📤 ToolResultBlock - Result length: {len(tool_result)} characters"
                                )
                                content += f"\n[ツール結果: {tool_result}]\n"
                            else:
                                if hasattr(block, "text"):
                                    content += block.text

                        except ImportError:
                            print(
                                "⚠️ Could not import specific block types, using fallback"
                            )
                            if hasattr(block, "text"):
                                content += block.text
                else:
                    content += str(message.content)

    except Exception as query_error:
        print(f"❌ Error during Claude Code SDK query: {query_error}")
        raise query_error

    print(
        f"✅ Query completed. Total messages: {message_count}, Content length: {len(content)}"
    )
    return content


def execute_websearch_fallback(search_queries: List[str]) -> str:
    """Execute WebSearch fallback when all parallel searches fail."""
    try:
        main_query = search_queries[0] if search_queries else ""
        print(f"🌐 Using WebSearch fallback for: {main_query}")

        websearch_prompt = f"""以下のクエリについて、WebSearchツールを使用して最新の情報を検索し、検索結果をまとめてください：

クエリ: {main_query}

要求事項:
- 最新の情報を検索してください
- 複数のソースから情報を収集してください
- 検索結果を日本語でまとめてください
- 信頼できる情報源を優先してください"""

        options = create_claude_code_options(
            "あなたは情報検索の専門家です。WebSearchツールを使用して、与えられたクエリに関する最新で正確な情報を検索し、結果をまとめてください。",
            max_turns=Config.CLAUDE_WEBSEARCH_MAX_TURNS,
        )

        return asyncio.run(execute_claude_code_query(websearch_prompt, options))

    except Exception as e:
        print(f"❌ WebSearch fallback failed: {e}")
        return f"WebSearch fallback error: {str(e)}"


def handle_claude_code_error(
    error_type: str, processed_output: str, error: Exception, state: WorkflowState
) -> WorkflowState:
    """Handle Claude Code SDK errors consistently."""
    import traceback

    print(f"🔍 Error type: {type(error)}")
    traceback.print_exc()

    error_message = f"{error_type}: {error}\n\n元の回答:\n{processed_output}"
    return {**state, "reviewed_output": error_message}
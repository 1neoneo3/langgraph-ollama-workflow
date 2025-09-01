"""Review node for Claude Code SDK integration."""

import asyncio

from ..core.state import WorkflowState
from ..services.review import (
    create_review_system_prompt,
    create_claude_code_options,
    execute_claude_code_query,
    handle_claude_code_error,
)
from ..utils.datetime_utils import get_current_datetime_info


def review_node(state: WorkflowState) -> WorkflowState:
    """Use Claude Code SDK to review and correct the final output."""
    processed_output = state.get("processed_output", "")
    original_question = state.get("original_user_input", "")

    if not processed_output:
        print("⚠️ No output to review")
        return {**state, "reviewed_output": ""}

    print("🔍 Reviewing output with Claude Code SDK...")
    print("📋 Starting Claude Code SDK review process...")

    try:
        print("📦 Importing Claude Code SDK...")
        print("✅ Claude Code SDK imported successfully")

        print("⚙️ Configuring Claude Code options with context7 MCP...")

        current_date_info = get_current_datetime_info()
        detailed_system_prompt = create_review_system_prompt(
            processed_output, original_question, current_date_info
        )

        options = create_claude_code_options(detailed_system_prompt)
        print("✅ Claude Code options configured with context7 MCP server")
        print(
            f"🔧 MCP servers configured: {list(options.mcp_servers.keys()) if options.mcp_servers else 'None'}"
        )
        print(f"🛠️ Allowed tools: {options.allowed_tools}")

        print("🔄 Starting async query to Claude Code SDK...")

        simple_prompt = "上記の回答内容を日本語で詳細にレビューしてください。すべての出力は必ず日本語で記述してください。"
        print(f"📏 Prompt length: {len(simple_prompt)} characters")

        print("🚀 Executing async query...")
        reviewed_content = asyncio.run(
            execute_claude_code_query(simple_prompt, options)
        )
        print("✅ Async query completed successfully")

        print("✅ Review completed with Claude Code SDK")
        print("-" * 60)
        print(reviewed_content)
        print("-" * 60)

        return {**state, "reviewed_output": reviewed_content}

    except ImportError as import_error:
        print(f"❌ Claude Code SDK not available: {import_error}")
        return handle_claude_code_error(
            "SDK利用不可", processed_output, import_error, state
        )
    except Exception as e:
        print(f"❌ Error during review: {e}")
        return handle_claude_code_error(
            "レビュー中にエラーが発生しました", processed_output, e, state
        )
#!/usr/bin/env python3
"""LangGraph workflow implementation with Ollama gpt-oss:20b model."""

from dotenv import load_dotenv
from src.main import main

# Load environment variables
load_dotenv()


# Configuration constants
class Config:
    # Ollama settings
    OLLAMA_MODEL = "gpt-oss:20b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_TEMPERATURE = 0.7

    # Search settings
    DEFAULT_SEARCH_DAYS_LIMIT = 60
    SEARCH_RESULT_LIMIT = 2000
    PARALLEL_SEARCH_LIMIT = 3
    SEARCH_TIMEOUT = 120
    INDIVIDUAL_RESULT_LIMIT = 1000

    # Time descriptions mapping
    TIME_DESCRIPTIONS = {
        1: "過去1日",
        7: "過去1週間",
        30: "過去1ヶ月",
        60: "過去2ヶ月",
        90: "過去3ヶ月",
        180: "過去6ヶ月",
        365: "過去1年",
    }

    # Recent search keywords
    RECENT_KEYWORDS = [
        "最新",
        "直近",
        "最近",
        "新しい",
        "今日",
        "今週",
        "今月",
        "latest",
        "recent",
        "new",
        "current",
        "today",
        "this week",
        "this month",
        "今年",
        "this year",
        "最新版",
        "最新バージョン",
        "current version",
        "latest version",
        "up to date",
        "アップデート",
        "update",
    ]

    # Time-specific keywords mapping
    TIME_SPECIFIC_KEYWORDS = {
        "今日": 1,
        "today": 1,
        "今週": 7,
        "this week": 7,
        "今月": 30,
        "this month": 30,
        "直近": 60,
        "最近": 60,
        "recent": 60,
    }

    # Slack settings
    SLACK_MAX_RETRIES = 3
    SLACK_INITIAL_RETRY_DELAY = 2
    SLACK_CONTENT_LIMIT = 3000

    # Claude Code settings
    CLAUDE_MAX_TURNS = 1
    CLAUDE_WEBSEARCH_MAX_TURNS = 3

    # Threading settings
    MAX_WORKERS = 3


# Define the state structure for our workflow
class WorkflowState(TypedDict):
    messages: list[BaseMessage]
    iteration: int
    user_input: str
    original_user_input: str  # Store original question for iterations
    processed_output: str
    should_continue: bool
    search_results: str
    search_queries: list[str]  # Store generated search queries
    parallel_search_stats: dict  # Store parallel search statistics
    recent_search_mode: bool
    search_days_limit: int  # Store specific time limit for search filtering
    initial_output: str  # Store first AI output for comparison
    reviewed_output: str  # Store Claude Code reviewed output
    document_generated: bool  # Track document generation status
    document_content: str  # Store generated markdown content
    document_path: str  # Store path to generated document
    slack_notification_sent: bool  # Track Slack notification status


# Helper functions
def get_current_datetime_info() -> Dict[str, any]:
    """Get current datetime information in a consistent format."""
    current_datetime = datetime.datetime.now()
    return {
        "datetime": current_datetime,
        "year": current_datetime.year,
        "month": current_datetime.month,
        "day": current_datetime.day,
        "date_str": current_datetime.strftime("%Y年%m月%d日"),
    }


def get_time_description(days: int) -> str:
    """Get human-readable time description for given days."""
    return Config.TIME_DESCRIPTIONS.get(days, f"過去{days}日")


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


def execute_psearch_with_progress(psearch_cmd: List[str]) -> Dict[str, any]:
    """Execute psearch command with real-time progress display."""
    import subprocess
    import sys

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


def search_node(state: WorkflowState) -> WorkflowState:
    """Search for relevant information using psearch with real-time output and enhanced date filtering."""
    user_input = state.get("user_input", "")
    recent_search_mode = state.get("recent_search_mode", False)
    search_days_limit = state.get("search_days_limit", Config.DEFAULT_SEARCH_DAYS_LIMIT)

    if not user_input:
        return {**state, "search_results": ""}

    current_date_info = get_current_datetime_info()

    search_mode_text = ""
    if recent_search_mode:
        time_desc = get_time_description(search_days_limit)
        search_mode_text = f" ({time_desc}の情報に限定)"

    print(f"🔍 Searching for information about: {user_input}{search_mode_text}")
    if recent_search_mode:
        time_desc = get_time_description(search_days_limit)
        print(
            f"📅 Date filtering active: {time_desc} ({current_date_info['date_str']}基準)"
        )
    print("📊 Progress visualization:")
    print("-" * 40)

    try:
        # Build and execute search command
        psearch_cmd = build_psearch_command(
            user_input, recent_search_mode, search_days_limit
        )

        # Show filtering info
        if recent_search_mode:
            time_desc = get_time_description(search_days_limit)
            if search_days_limit <= 30:
                print(f"📅 Filtering results: {time_desc}以内, 日付順ソート")
            else:
                months = max(1, search_days_limit // 30)
                print(f"📅 Filtering results: 過去{months}ヶ月以内, 日付順ソート")

        # Execute search with progress display
        result = execute_psearch_with_progress(psearch_cmd)

        print("-" * 40)
        print(f"⏱️ Search completed in {result['elapsed_time']:.2f} seconds")

        if result["success"]:
            print("✅ Search completed successfully")
            search_output = result["stdout"]
            result_count = (
                len(search_output.split("---")) - 1
                if "---" in search_output
                else "some"
            )
            print(f"📄 Found {result_count} results")

            # Summarize search results for LLM processing
            search_summary = f"Search results for '{user_input}':\n\n{search_output[: Config.SEARCH_RESULT_LIMIT]}..."
            return {**state, "search_results": search_summary}
        else:
            print(f"⚠️ Search failed with return code {result['return_code']}")
            print(f"Error: {result['stderr']}")
            return {**state, "search_results": f"Search failed: {result['stderr']}"}

    except FileNotFoundError:
        print("❌ psearch command not found")
        return {**state, "search_results": "psearch command not available"}
    except Exception as e:
        print(f"❌ Search error: {e}")
        return {**state, "search_results": f"Search error: {str(e)}"}


def detect_recent_search_mode(
    user_input: str, current_date_info: Dict[str, any]
) -> tuple[bool, int]:
    """Detect if recent search mode should be activated and determine time limit."""
    # Enhanced keywords including dynamic current year
    recent_keywords = Config.RECENT_KEYWORDS + [
        f"{current_date_info['year']}年",
        f"{current_date_info['year'] - 1}年",
    ]

    recent_search_mode = any(keyword in user_input for keyword in recent_keywords)

    # Determine specific time range
    search_days_limit = Config.DEFAULT_SEARCH_DAYS_LIMIT
    for keyword, days in Config.TIME_SPECIFIC_KEYWORDS.items():
        if keyword in user_input:
            search_days_limit = min(search_days_limit, days)
            break

    if recent_search_mode:
        time_description = get_time_description(search_days_limit)
        print(
            f"🔍 Recent information keywords detected - search will be limited to {time_description}"
        )
        filter_year = current_date_info["year"] - (1 if search_days_limit > 30 else 0)
        print(
            f"📅 Current date: {current_date_info['date_str']} - filtering for content from {filter_year}年以降"
        )

    return recent_search_mode, search_days_limit


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


def create_ollama_llm():
    """Create and return configured Ollama LLM instance."""
    return ChatOllama(
        model=Config.OLLAMA_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        temperature=Config.OLLAMA_TEMPERATURE,
    )


def handle_ollama_fallback(
    messages: List[BaseMessage], iteration: int
) -> Dict[str, any]:
    """Handle Ollama fallback when service is unavailable."""
    if messages and isinstance(messages[-1], HumanMessage):
        content = messages[-1].content
        fallback_response = (
            f"Processing iteration {iteration}: {content} (Ollama unavailable)"
        )
        messages.append(AIMessage(content=fallback_response))

        return {
            "messages": messages,
            "processed_output": fallback_response,
        }
    return {"messages": messages}


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

        import asyncio

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


def handle_claude_code_error(
    error_type: str, processed_output: str, error: Exception, state: WorkflowState
) -> WorkflowState:
    """Handle Claude Code SDK errors consistently."""
    import traceback

    print(f"🔍 Error type: {type(error)}")
    traceback.print_exc()

    error_message = f"{error_type}: {error}\n\n元の回答:\n{processed_output}"
    return {**state, "reviewed_output": error_message}


def documentation_node(state: WorkflowState) -> WorkflowState:
    """Generate markdown documentation comparing initial and final outputs."""
    original_question = state.get("original_user_input", "")
    initial_output = state.get("initial_output", "")
    reviewed_output = state.get("reviewed_output", "")
    search_results = state.get("search_results", "")

    print("📝 Generating documentation...")

    try:
        # Create docs directory if it doesn't exist
        docs_dir = Path.home() / "workspace" / "Docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Create a descriptive title from the original question
        question_summary = (
            original_question[:30]
            .replace("/", "")
            .replace("\\", "")
            .replace(":", "：")
            .replace("?", "？")
            .replace("*", "")
            .replace("<", "")
            .replace(">", "")
            .replace("|", "")
        )
        if len(original_question) > 30:
            question_summary += "..."

        filename = f"{question_summary}_分析結果.md"
        file_path = docs_dir / filename

        # Extract final corrected version if available
        final_corrected_version = ""
        if reviewed_output:
            try:
                # Try to extract corrected content from review output
                # Look for patterns like "修正版:" or actual corrected text sections
                import re

                # Look for corrected text in the review output with more comprehensive patterns
                corrected_patterns = [
                    r"修正版[：:]\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
                    r"修正[：:]\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
                    r"改善版[：:]\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
                    r"以下が修正版です[：:]?\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
                    r"修正後[：:]?\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
                ]

                for pattern in corrected_patterns:
                    match = re.search(
                        pattern, reviewed_output, re.DOTALL | re.MULTILINE
                    )
                    if match:
                        final_corrected_version = match.group(1).strip()
                        print(
                            f"✅ Extracted corrected version using pattern: {pattern[:20]}..."
                        )
                        break

                # If no explicit corrected version found, check if the review contains substantial corrections
                # Look for structured corrections or improvements
                if not final_corrected_version:
                    # Check for markdown-style corrections or improvements
                    improvement_patterns = [
                        r"## レビュー結果.*?## 修正内容.*?\n(.+?)(?=\n## |$)",
                        r"### 修正内容\s*\n(.+?)(?=\n### |$)",
                        r"**修正版**\s*\n(.+?)(?=\n**|$)",
                        r"\*\*修正版\*\*\s*\n(.+?)(?=\n\*\*|$)",
                    ]

                    for pattern in improvement_patterns:
                        match = re.search(
                            pattern, reviewed_output, re.DOTALL | re.MULTILINE
                        )
                        if match:
                            final_corrected_version = match.group(1).strip()
                            print("✅ Extracted improvement section using pattern")
                            break

                # If still no corrected version, check if the review contains substantial content that looks like a correction
                if (
                    not final_corrected_version
                    and "修正" in reviewed_output
                    and len(reviewed_output) > 1000
                ):
                    # Check if the review output seems to contain a complete corrected version
                    # Look for technical content or structured information
                    if any(
                        keyword in reviewed_output
                        for keyword in [
                            "Linear",
                            "GitHub",
                            "機能",
                            "実装",
                            "設定",
                            "手順",
                        ]
                    ):
                        print(
                            "✅ Using complete review output as it contains substantial technical corrections"
                        )
                        final_corrected_version = reviewed_output

            except Exception as e:
                print(f"⚠️ Could not extract corrected version: {e}")

        # Generate markdown content
        markdown_content = f"""# LangGraphワークフロー実行結果

## 実行情報
- **実行日時**: {datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
- **質問**: {original_question}
- **ワークフローイテレーション**: {state.get("iteration", 0)}

## 元の質問
```
{original_question}
```

## 検索結果の概要
```
{search_results[:500] if search_results else "検索結果なし"}...
```

## 1. 初回AI回答（Ollama gpt-oss:20b）
{initial_output if initial_output else "初回回答なし"}

## 2. Claude Codeレビュー・修正結果
{reviewed_output if reviewed_output else "レビュー結果なし"}

{
            f'''## 3. 最終修正版

以下はClaude Codeレビューに基づく修正版です：

{final_corrected_version}

### 修正の詳細説明
上記の修正版は元の回答に対するレビューで指摘された以下の改善点を反映しています：
- 技術的正確性の向上
- 最新情報の追加
- 論理的一貫性の改善
- 完全性の向上
'''
            if final_corrected_version and final_corrected_version != reviewed_output
            else ""
        }


---
*このドキュメントは LangGraph + Claude Code SDK ワークフローにより自動生成されました*
"""

        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✅ Documentation generated: {file_path}")

        return {
            **state,
            "document_generated": True,
            "document_content": markdown_content,
            "document_path": str(file_path),
        }

    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        return {
            **state,
            "document_generated": False,
            "document_content": "",
            "document_path": "",
        }


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
        import asyncio

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
            queries = [
                user_input,  # Basic query
                f"{user_input} 最新",  # Latest info query
                f"{user_input} 実装",  # Implementation query
            ][:3]

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
        # Fallback: create exactly 3 basic queries from user input
        fallback_queries = [
            user_input,  # Basic query
            f"{user_input} 最新",  # Latest info query
            f"{user_input} 実装方法",  # Implementation query
        ]
        return {**state, "search_queries": fallback_queries}

    except Exception as e:
        print(f"❌ Error with Claude Code agent: {e}")
        # Fallback: create exactly 3 basic queries from user input
        fallback_queries = [
            user_input,  # Basic query
            f"{user_input} 最新",  # Latest info query
            f"{user_input} 実装方法",  # Implementation query
        ]
        return {**state, "search_queries": fallback_queries}


def execute_single_search(
    query_info: tuple, recent_search_mode: bool, search_days_limit: int
) -> Dict[str, any]:
    """Execute a single search with proper error handling."""
    import subprocess

    query_index, query = query_info
    print(f"🔎 Search {query_index + 1}: {query}")

    try:
        # Build psearch command using existing helper
        psearch_cmd = build_psearch_command(
            query, recent_search_mode, search_days_limit
        )
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

        import asyncio

        return asyncio.run(execute_claude_code_query(websearch_prompt, options))

    except Exception as e:
        print(f"❌ WebSearch fallback failed: {e}")
        return f"WebSearch fallback error: {str(e)}"


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


def parallel_search_node(state: WorkflowState) -> WorkflowState:
    """Execute multiple searches in parallel using the generated queries."""
    search_queries = state.get("search_queries", [])
    recent_search_mode = state.get("recent_search_mode", False)
    search_days_limit = state.get("search_days_limit", Config.DEFAULT_SEARCH_DAYS_LIMIT)

    if not search_queries:
        print("⚠️ No search queries available")
        return {**state, "search_results": ""}

    print(f"🔍 Executing {len(search_queries)} parallel searches...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

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


def validate_slack_webhook_url(webhook_url: str) -> tuple[bool, str]:
    """Validate Slack webhook URL format."""
    if not webhook_url:
        return False, "SLACK_WEBHOOK_URL not found in environment variables"

    if not webhook_url.startswith("https://hooks.slack.com/"):
        return False, f"Invalid Slack webhook URL format: {webhook_url[:50]}..."

    return True, f"Slack webhook URL validated: {webhook_url[:30]}..."


def create_slack_payload(
    document_content: str, document_path: str, original_question: str
) -> Dict[str, any]:
    """Create Slack payload based on content size."""
    if len(document_content) > Config.SLACK_CONTENT_LIMIT:
        print(
            f"📄 Large content detected ({len(document_content)} chars), using simplified format"
        )

        summary = f"""
📄 LangGraphワークフロー実行完了

質問: {original_question}
ドキュメント生成パス: {document_path}

内容が大きいため、完全な結果は生成されたドキュメントファイルをご確認ください。

実行結果概要:
- Ollamaによる初期回答生成完了
- Claude Codeによるレビューと事実確認完了
- 最新情報との照合完了
- ドキュメント生成完了
        """.strip()

        return {
            "text": summary,
            "username": "LangGraph Workflow Bot",
            "icon_emoji": ":memo:",
        }
    else:
        return {
            "text": f"""📄 LangGraphワークフロー実行完了

質問: {original_question}
ドキュメント生成パス: `{document_path}`

結果:
```
{document_content}
```""",
            "username": "LangGraph Workflow Bot",
            "icon_emoji": ":memo:",
        }


def send_slack_message_with_retry(
    webhook_url: str, payload: Dict[str, any], document_content: str
) -> bool:
    """Send Slack message with retry mechanism."""
    import requests
    import json

    retry_delay = Config.SLACK_INITIAL_RETRY_DELAY

    for attempt in range(Config.SLACK_MAX_RETRIES):
        try:
            print(f"🔄 送信試行 {attempt + 1}/{Config.SLACK_MAX_RETRIES}")

            start_time = time.time()
            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                print("✅ Slack notification sent successfully")
                print(f"📊 Document content size: {len(document_content)} characters")
                print(f"⏱️ Response time: {response_time:.2f} seconds")
                return True
            else:
                print(f"❌ Slack notification failed: {response.status_code}")
                print(f"📄 Response: {response.text}")

                # Don't retry for client errors
                if response.status_code in [400, 404]:
                    if response.status_code == 400:
                        print("💡 Bad Request - チェックポイント:")
                        print("  - Webhook URLが正しいか確認してください")
                    elif response.status_code == 404:
                        print("💡 Not Found - Webhook URLが無効または削除されています")
                    return False

                if attempt < Config.SLACK_MAX_RETRIES - 1:
                    print(f"⏳ {retry_delay}秒後にリトライします...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
        ) as e:
            error_type = type(e).__name__
            print(
                f"📡 {error_type} (attempt {attempt + 1}/{Config.SLACK_MAX_RETRIES}): {e}"
            )

            if attempt < Config.SLACK_MAX_RETRIES - 1:
                print(f"⏳ {retry_delay}秒後にリトライします...")
                time.sleep(retry_delay)
                retry_delay *= 2

    print(f"❌ Slack notification failed after {Config.SLACK_MAX_RETRIES} attempts")
    return False


def slack_notification_node(state: WorkflowState) -> WorkflowState:
    """Send Slack notification with the complete document content and retry mechanism."""
    document_content = state.get("document_content", "")
    document_path = state.get("document_path", "")
    original_question = state.get("original_user_input", "")

    if not document_content:
        print("⚠️ No document content available for Slack notification")
        return {**state, "slack_notification_sent": False}

    print("📢 Sending Slack notification with document content...")

    try:
        # Get and validate Slack webhook URL
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        is_valid, validation_message = validate_slack_webhook_url(slack_webhook_url)

        if not is_valid:
            print(f"⚠️ {validation_message}")
            if "not found" in validation_message:
                print(
                    "💡 設定方法: export SLACK_WEBHOOK_URL='https://hooks.slack.com/your/webhook/url'"
                )
            elif "Invalid" in validation_message:
                print("💡 正しい形式: https://hooks.slack.com/services/...")
            return {**state, "slack_notification_sent": False}

        print(f"✅ {validation_message}")

        # Create payload and send message
        slack_payload = create_slack_payload(
            document_content, document_path, original_question
        )
        success = send_slack_message_with_retry(
            slack_webhook_url, slack_payload, document_content
        )

        return {**state, "slack_notification_sent": success}

    except ImportError:
        print("❌ requests library not available for Slack notification")
        print("💡 Install with: pip install requests")
        return {**state, "slack_notification_sent": False}
    except Exception as e:
        print(f"❌ Unexpected error sending Slack notification: {e}")
        import traceback

        traceback.print_exc()
        return {**state, "slack_notification_sent": False}


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow with Ollama."""

    # Create the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes to the workflow
    workflow.add_node("input", input_node)
    workflow.add_node("query_generation", generate_search_queries)
    workflow.add_node("parallel_search", parallel_search_node)
    workflow.add_node("process", processing_node)
    workflow.add_node("review", review_node)
    workflow.add_node("document", documentation_node)
    # Check if Slack webhook URL is configured
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        workflow.add_node("slack_notification", slack_notification_node)

    # Define the workflow edges with conditional Slack notification
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "query_generation")
    workflow.add_edge("query_generation", "parallel_search")
    workflow.add_edge("parallel_search", "process")
    workflow.add_edge("process", "review")
    workflow.add_edge("review", "document")

    if slack_webhook_url:
        workflow.add_edge("document", "slack_notification")
        workflow.add_edge("slack_notification", END)
    else:
        workflow.add_edge("document", END)

    return workflow


def check_ollama_connection() -> bool:
    """Check if Ollama is running and the configured model is available."""
    try:
        import requests

        print("🔍 Checking Ollama connection...")

        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)

        if response.status_code != 200:
            print(f"❌ Ollama API returned error: {response.status_code}")
            return False

        models = response.json()
        model_names = [model["name"] for model in models.get("models", [])]

        print(f"✅ Ollama is running with {len(model_names)} models")

        if Config.OLLAMA_MODEL in model_names:
            print(f"✅ {Config.OLLAMA_MODEL} model is available")
            return True
        else:
            print(f"❌ {Config.OLLAMA_MODEL} model not found")
            print("Available models:", model_names)
            print(
                f"\n💡 To install {Config.OLLAMA_MODEL}, run: ollama pull {Config.OLLAMA_MODEL}"
            )
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\n💡 Make sure Ollama is running: ollama serve")
        return False
    except ImportError:
        print("❌ requests library not available for Ollama check")
        return False


def main():
    """Main function to run the workflow with Ollama."""
    print("🚀 Starting LangGraph Workflow with Ollama gpt-oss:20b")
    print("=" * 60)

    # Check Ollama connection
    ollama_available = check_ollama_connection()
    if not ollama_available:
        print("\n⚠️  Continuing anyway - will use fallback responses if needed")

    print()

    # Get user input
    print("💬 Please enter your question:")
    try:
        user_question = input("❓ ")
    except EOFError:
        user_question = ""

    if not user_question.strip():
        user_question = "Explain the concept of LangGraph workflows and their benefits for AI applications"
        print(f"🔄 Using default question: {user_question}")

    # Create the workflow
    workflow = create_workflow()

    # Compile the workflow
    app = workflow.compile()

    # Initial state
    initial_state = {
        "messages": [],
        "iteration": 0,
        "user_input": user_question,
        "original_user_input": user_question,  # Store original question
        "processed_output": "",
        "should_continue": True,
        "search_results": "",
        "search_queries": [],  # Store generated search queries
        "parallel_search_stats": {},  # Store parallel search statistics
        "recent_search_mode": False,
        "search_days_limit": 60,  # Default to 2 months
        "initial_output": "",  # Store first AI output for comparison
        "reviewed_output": "",  # Store Claude Code reviewed output
        "document_generated": False,  # Track document generation status
        "document_content": "",  # Store generated markdown content
        "document_path": "",  # Store path to generated document
        "slack_notification_sent": not bool(
            os.getenv("SLACK_WEBHOOK_URL")
        ),  # Track Slack notification status (True if not needed)
    }

    print("\n📋 Initial State:")
    print(f"  User Input: {initial_state['user_input']}")
    print(f"  Iteration: {initial_state['iteration']}")
    print()

    # Execute the workflow
    print("⚡ Executing Workflow with Ollama...")
    print("-" * 40)

    try:
        final_state = app.invoke(initial_state)

        print("\n✅ Workflow Completed!")
        print("=" * 60)
        print("📊 Final Results:")
        print(f"  Total Iterations: {final_state['iteration']}")
        print(f"  Message Count: {len(final_state['messages'])}")
        print(
            f"  Document Generated: {'✅' if final_state.get('document_generated', False) else '❌'}"
        )
        print(
            f"  Slack Notification: {'✅' if final_state.get('slack_notification_sent', False) else '❌'}"
        )
        print()

        print("💬 Full Conversation History:")
        for i, message in enumerate(final_state["messages"], 1):
            message_type = (
                "User" if isinstance(message, HumanMessage) else "AI (gpt-oss:20b)"
            )
            content = message.content
            print(f"  {i}. [{message_type}]:")
            print("-" * 50)
            print(content)
            print("-" * 50)
            print()

        # Display review results if available
        if final_state.get("reviewed_output"):
            print("🔍 Claude Code Review Results:")
            print("=" * 60)
            print(final_state["reviewed_output"])
            print("=" * 60)
            print()

        # Display documentation status
        if final_state.get("document_generated"):
            print("📝 Documentation successfully generated in Docs/ directory")
            print(f"📄 Document path: {final_state.get('document_path', 'Unknown')}")
        else:
            print("⚠️ Documentation generation failed or was skipped")

        # Display Slack notification status
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook_url:
            if final_state.get("slack_notification_sent"):
                print(
                    "📢 Slack notification sent successfully with complete document content"
                )
            else:
                print(
                    "⚠️ Slack notification failed (check SLACK_WEBHOOK_URL environment variable)"
                )
        else:
            print("ℹ️ Slack notification skipped (SLACK_WEBHOOK_URL not configured)")

    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

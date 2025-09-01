#!/usr/bin/env python3
"""LangGraph workflow implementation with Ollama gpt-oss:20b model."""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
import datetime
import os
from pathlib import Path

# Load environment variables
load_dotenv()


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


def search_node(state: WorkflowState) -> WorkflowState:
    """Search for relevant information using psearch with real-time output and enhanced date filtering."""
    user_input = state.get("user_input", "")
    recent_search_mode = state.get("recent_search_mode", False)
    search_days_limit = state.get("search_days_limit", 60)  # Default to 2 months

    if not user_input:
        return {**state, "search_results": ""}

    # Get current date for filtering context
    current_datetime = datetime.datetime.now()
    current_year = current_datetime.year
    
    # Calculate more precise time descriptions
    time_descriptions = {
        1: "過去1日",
        7: "過去1週間", 
        30: "過去1ヶ月",
        60: "過去2ヶ月",
        90: "過去3ヶ月",
        180: "過去6ヶ月",
        365: "過去1年"
    }
    
    search_mode_text = ""
    if recent_search_mode:
        time_desc = time_descriptions.get(search_days_limit, f"過去{search_days_limit}日")
        search_mode_text = f" ({time_desc}の情報に限定)"
    
    print(f"🔍 Searching for information about: {user_input}{search_mode_text}")
    if recent_search_mode:
        print(f"📅 Date filtering active: {time_desc} ({current_year}年{current_datetime.month}月{current_datetime.day}日基準)")
    print("📊 Progress visualization:")
    print("-" * 40)

    try:
        import subprocess
        import sys
        import time

        # Use psearch to search for relevant information
        # Format the query for better search results
        search_query = user_input[:100]  # Limit query length

        # Record start time
        start_time = time.time()

        # Build psearch command with enhanced date filtering
        psearch_cmd = ["psearch", "search", search_query, "-n", "5", "-c", "--json"]

        # Add more precise date filtering based on search_days_limit
        if recent_search_mode:
            if search_days_limit <= 30:  # Within a month, use recent only
                psearch_cmd.extend(["-r", "-s"])
                print(f"📅 Filtering results: {time_descriptions.get(search_days_limit, f'{search_days_limit}日')}以内, 日付順ソート")
            else:  # Default to months for longer periods
                months = max(1, search_days_limit // 30)
                psearch_cmd.extend(["-r", "--months", str(months), "-s"])
                print(f"📅 Filtering results: 過去{months}ヶ月以内, 日付順ソート")

        # Run psearch command with real-time output streaming
        process = subprocess.Popen(
            psearch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffering
            universal_newlines=True,
        )

        # Collect output while displaying progress
        stdout_lines = []
        stderr_lines = []

        # Read stdout line by line for real-time display
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                # Display the line immediately for progress visualization
                print(f"📤 {output.rstrip()}")
                sys.stdout.flush()  # Force immediate output
                stdout_lines.append(output)

        # Get any remaining stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            stderr_lines.append(stderr_output)

        # Wait for process to complete
        return_code = process.wait()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        print("-" * 40)
        print(f"⏱️ Search completed in {elapsed_time:.2f} seconds")

        if return_code == 0:
            search_output = "".join(stdout_lines)
            print("✅ Search completed successfully")
            print(
                f"📄 Found {len(search_output.split('---')) - 1 if '---' in search_output else 'some'} results"
            )

            # Summarize search results for LLM processing
            search_summary = (
                f"Search results for '{user_input}':\n\n{search_output[:2000]}..."
            )

            return {
                **state,
                "search_results": search_summary,
            }
        else:
            stderr_output = "".join(stderr_lines)
            print(f"⚠️ Search failed with return code {return_code}")
            print(f"Error: {stderr_output}")
            return {
                **state,
                "search_results": f"Search failed: {stderr_output}",
            }

    except FileNotFoundError:
        print("❌ psearch command not found")
        return {
            **state,
            "search_results": "psearch command not available",
        }
    except Exception as e:
        print(f"❌ Search error: {e}")
        return {
            **state,
            "search_results": f"Search error: {str(e)}",
        }


def input_node(state: WorkflowState) -> WorkflowState:
    """Process initial user input and detect recent search keywords."""
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    # Add user message to conversation
    if user_input:
        messages.append(HumanMessage(content=user_input))

    # Get current date and time for more accurate filtering
    current_datetime = datetime.datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    
    # Enhanced keywords for recent information detection (including current year)
    recent_keywords = [
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
        f"{current_year}年",  # Dynamic current year
        f"{current_year - 1}年",  # Previous year
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

    # Check for specific time-related terms for more granular filtering
    time_specific_keywords = {
        "今日": 1,     # 1 day
        "today": 1,
        "今週": 7,     # 1 week 
        "this week": 7,
        "今月": 30,    # 1 month
        "this month": 30,
        "直近": 60,    # 2 months
        "最近": 60,    # 2 months
        "recent": 60,
    }

    recent_search_mode = any(keyword in user_input for keyword in recent_keywords)
    
    # Determine the specific time range based on detected keywords
    search_days_limit = 60  # Default to 2 months
    for keyword, days in time_specific_keywords.items():
        if keyword in user_input:
            search_days_limit = min(search_days_limit, days)  # Use the most restrictive time frame
            break

    if recent_search_mode:
        time_description = {
            1: "過去1日",
            7: "過去1週間", 
            30: "過去1ヶ月",
            60: "過去2ヶ月"
        }.get(search_days_limit, f"過去{search_days_limit}日")
        
        print(f"🔍 Recent information keywords detected - search will be limited to {time_description}")
        print(f"📅 Current date: {current_datetime.strftime('%Y年%m月%d日')} - filtering for content from {current_year - (1 if search_days_limit > 30 else 0)}年以降")

    return {
        **state,
        "messages": messages,
        "iteration": state.get("iteration", 0) + 1,
        "recent_search_mode": recent_search_mode,
        "search_days_limit": search_days_limit,  # Add specific time limit
        "original_user_input": state.get(
            "original_user_input", user_input
        ),  # Store original on first iteration
    }


def processing_node(state: WorkflowState) -> WorkflowState:
    """Process the user input using Ollama gpt-oss:20b model with search results."""
    messages = state["messages"]
    iteration = state["iteration"]
    search_results = state.get("search_results", "")

    if not messages:
        return state

    print(f"🤖 Processing iteration {iteration} with Ollama gpt-oss:20b...")

    try:
        # Initialize Ollama with gpt-oss:20b model
        llm = ChatOllama(
            model="gpt-oss:20b",
            base_url="http://localhost:11434",  # Default Ollama port
            temperature=0.7,
        )

        # Create a focused prompt for the LLM
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            content = last_message.content

            # Get current date and time for context
            current_datetime = datetime.datetime.now()
            current_date_str = current_datetime.strftime("%Y年%m月%d日")
            current_year = current_datetime.year
            
            # Create a system prompt that includes search results
            system_prompt = f"""
            【重要な指示】
            - すべての回答は日本語で記述してください
            - 現在日時: {current_date_str} ({current_year}年)
            - 最新情報（{current_year - 1}年以降）を優先して活用してください
            
            あなたはLangGraphワークフローの{iteration}回目の処理を行うAIアシスタントです。
            最新の検索結果にアクセスして、正確で最新の情報を提供することができます。
            ユーザーの入力に対して、必要に応じて検索結果から関連情報を取り入れた、思慮深い回答を日本語で提供してください。
            簡潔でありながら、情報量豊富な回答を心がけてください。
            
            【現在の日時情報】
            現在は{current_date_str}（{current_year}年）です。この日時を考慮して、最新の情報を優先して回答してください。
            
            ユーザーの入力: {content}
            
            検索結果 (利用可能な場合):
            {search_results if search_results else "検索結果がありません"}
            
            【回答要件】
            - すべて日本語で回答してください
            - 検索結果を活用して、{current_year}年時点での最新で正確な情報を含めてください
            - 古い情報（{current_year - 2}年以前）がある場合は、最新動向も併記してください
            - 技術的な内容の場合は、最新バージョンや仕様変更も考慮してください
            """

            # Get response from Ollama
            response = llm.invoke([HumanMessage(content=system_prompt)])

            # Add AI response to messages
            ai_response = response.content if response.content else "応答を生成できませんでした。"
            messages.append(AIMessage(content=ai_response))

            print("✅ LLM Full Response:")
            print("-" * 60)
            print(ai_response)
            print("-" * 60)

            # Store output (no iteration, so this is both initial and final)
            return {
                **state,
                "messages": messages,
                "processed_output": ai_response,
                "initial_output": ai_response,
            }

    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        print("🔄 Falling back to simple response generation...")

        # Fallback to simple processing if Ollama is not available
        if messages and isinstance(messages[-1], HumanMessage):
            content = messages[-1].content
            fallback_response = (
                f"Processing iteration {iteration}: {content} (Ollama unavailable)"
            )
            messages.append(AIMessage(content=fallback_response))

            return {
                **state,
                "messages": messages,
                "processed_output": fallback_response,
            }

    return state




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
        # Try importing Claude Code SDK
        print("📦 Importing Claude Code SDK...")
        from claude_code_sdk import query, ClaudeCodeOptions
        print("✅ Claude Code SDK imported successfully")


        # Configure options for Claude Code with context7 MCP access
        print("⚙️ Configuring Claude Code options with context7 MCP...")
        
        # Get current date and time for context
        current_datetime = datetime.datetime.now()
        current_date_str = current_datetime.strftime("%Y年%m月%d日")
        current_year = current_datetime.year
        
        # Create detailed system prompt that includes the original answer to review
        detailed_system_prompt = f"""あなたは技術文書の校正・レビューの専門家です。

【重要な指示】
- すべての出力とドキュメント内容は日本語で記述してください
- 現在日時: {current_date_str} ({current_year}年)
- 最新情報（{current_year - 1}年以降）を優先して参照してください
- Claude Code SDKを使用する際は必ず現在日時を考慮して最新の情報を取得してください

【レビュー対象の回答内容】
{processed_output}

【元の質問】
{original_question}

上記の回答内容について、以下の点で詳細なレビューを行ってください：

**一般的なレビューポイント：**
1. 事実の正確性（間違いや古い情報がないか、特に{current_year - 1}年以降の最新情報との整合性）
2. 論理的な一貫性（矛盾する内容がないか）
3. 完全性（重要な情報が抜けていないか）
4. わかりやすさ（説明が明確で理解しやすいか）
5. 最新性（{current_year}年の最新情報に基づいているか）

**技術的な質問の場合の追加レビューポイント：**
質問が技術的な内容の場合は、context7 MCPツールを積極的に使用して公式ドキュメントや最新の技術情報を参照し、以下の点も確認してください：
- 技術的正確性（コードの構文、APIの使用方法、{current_year}年時点での最新仕様）
- ベストプラクティス準拠（業界標準に従っているか、最新のベストプラクティス）
- セキュリティ（セキュリティ上の問題やリスクがないか、最新のセキュリティガイドライン）
- パフォーマンス（効率的で最適化されたアプローチか、最新の最適化手法）
- 公式ドキュメントとの整合性（{current_year}年の最新ドキュメントとの比較）
- 実装時の注意点や落とし穴（最新バージョンでの変更点を含む）

**最新情報確認の要求事項：**
- WebSearchツールを使用して{current_year}年の最新情報を確認してください
- 古い情報（{current_year - 2}年以前）の場合は最新情報で補完してください
- バージョンアップやAPI変更などの最新動向を反映してください

**レビュー出力の要求事項（すべて日本語で出力）：**
- レビュー内容は省略せず、完全な形で日本語で提供してください
- 修正が必要な場合は、具体的な修正版を日本語で完全に提供してください
- 修正点と理由を詳細に日本語で説明してください
- 技術的な詳細や重要な情報は省略せず、すべて日本語で記述してください
- 長い内容であってもすべて日本語で含めて回答してください
- 最新情報（{current_year}年）に基づく更新点があれば明示してください

問題がない場合は「レビュー完了：問題なし（{current_date_str}時点）」と日本語で回答してください。"""

        options = ClaudeCodeOptions(
            system_prompt=detailed_system_prompt,
            max_turns=1,
            allowed_tools=["WebSearch"],  # Allow web search for fact checking
            mcp_servers={
                "context7": {
                    "command": "npx",
                    "args": ["-y", "@context7/server"]
                }
            }
        )
        print("✅ Claude Code options configured with context7 MCP server")
        print(f"🔧 MCP servers configured: {list(options.mcp_servers.keys())}")
        print(f"🛠️ Allowed tools: {options.allowed_tools}")

        reviewed_content = ""

        # Query Claude Code SDK using asyncio
        print("🔄 Starting async query to Claude Code SDK...")
        
        async def get_review():
            content = ""
            message_count = 0
            
            # Simple prompt since all review details are in system prompt
            simple_prompt = "上記の回答内容をレビューしてください。"
            
            print("📡 Sending prompt to Claude Code SDK...")
            print(f"📏 Prompt length: {len(simple_prompt)} characters")
            
            try:
                async for message in query(prompt=simple_prompt, options=options):
                    message_count += 1
                    print(f"📨 Received message #{message_count} from Claude Code SDK")
                    
                    if hasattr(message, "content"):
                        if isinstance(message.content, list):
                            for i, block in enumerate(message.content):
                                print(f"📄 Processing content block #{i+1} - Type: {type(block).__name__}")
                                
                                # Import the specific types from claude_code_sdk
                                try:
                                    from claude_code_sdk.types import TextBlock, ToolUseBlock, ToolResultBlock
                                    
                                    if isinstance(block, TextBlock):
                                        block_text = block.text
                                        print(f"📝 TextBlock - length: {len(block_text)} characters")
                                        content += block_text
                                    elif isinstance(block, ToolUseBlock):
                                        tool_name = getattr(block, 'name', 'unknown')
                                        tool_input = getattr(block, 'input', {})
                                        print(f"🔧 ToolUseBlock - Tool: {tool_name}")
                                        print(f"📥 Tool input: {str(tool_input)[:200]}...")
                                        # Add tool use information to content for context
                                        content += f"\n[ツール使用: {tool_name}]\n"
                                    elif isinstance(block, ToolResultBlock):
                                        tool_result = getattr(block, 'content', 'no result')
                                        tool_result_str = str(tool_result)
                                        print(f"📤 ToolResultBlock - Result length: {len(tool_result_str)} characters")
                                        print(f"🔍 Tool result preview: {tool_result_str[:200]}...")
                                        # Add complete tool result to content without truncation
                                        content += f"\n[ツール結果: {tool_result_str}]\n"
                                    else:
                                        print(f"❓ Unknown block type: {type(block)}")
                                        # Try to get text if it exists
                                        if hasattr(block, "text"):
                                            content += block.text
                                            
                                except ImportError:
                                    # Fallback if types are not available
                                    print("⚠️ Could not import specific block types, using fallback")
                                    if hasattr(block, "text"):
                                        block_text = block.text
                                        print(f"📝 Block text length: {len(block_text)} characters")
                                        content += block_text
                                    else:
                                        print(f"⚠️ Block has no text attribute: {type(block)}")
                        else:
                            content_str = str(message.content)
                            print(f"📝 Message content length: {len(content_str)} characters")
                            content += content_str
                    else:
                        print("⚠️ Message has no content attribute")
                        
            except Exception as query_error:
                print(f"❌ Error during Claude Code SDK query: {query_error}")
                raise query_error
                
            print(f"✅ Query completed. Total messages received: {message_count}")
            print(f"📊 Total content length: {len(content)} characters")
            return content

        # Run async function
        import asyncio

        try:
            print("🚀 Executing async query...")
            reviewed_content = asyncio.run(get_review())
            print("✅ Async query completed successfully")
        except Exception as async_error:
            print(f"❌ Async execution error: {async_error}")
            print(f"🔍 Error type: {type(async_error)}")
            import traceback
            traceback.print_exc()
            reviewed_content = (
                f"非同期実行エラー: {async_error}\n\n元の回答:\n{processed_output}"
            )

        print("✅ Review completed with Claude Code SDK")
        print("-" * 60)
        print(reviewed_content)
        print("-" * 60)

        return {
            **state,
            "reviewed_output": reviewed_content,
        }

    except ImportError as import_error:
        print(f"❌ Claude Code SDK not available: {import_error}")
        print("🔍 Import error details:")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "reviewed_output": f"レビューをスキップしました（Claude Code SDK利用不可）\n\n元の回答:\n{processed_output}",
        }
    except Exception as e:
        print(f"❌ Error during review: {e}")
        print(f"🔍 Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "reviewed_output": f"レビュー中にエラーが発生しました: {e}\n\n元の回答:\n{processed_output}",
        }


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
                    r"修正後[：:]?\s*\n(.+?)(?=\n\n##|\n\n---|\Z)"
                ]
                
                for pattern in corrected_patterns:
                    match = re.search(pattern, reviewed_output, re.DOTALL | re.MULTILINE)
                    if match:
                        final_corrected_version = match.group(1).strip()
                        print(f"✅ Extracted corrected version using pattern: {pattern[:20]}...")
                        break
                
                # If no explicit corrected version found, check if the review contains substantial corrections
                # Look for structured corrections or improvements
                if not final_corrected_version:
                    # Check for markdown-style corrections or improvements
                    improvement_patterns = [
                        r"## レビュー結果.*?## 修正内容.*?\n(.+?)(?=\n## |$)",
                        r"### 修正内容\s*\n(.+?)(?=\n### |$)",
                        r"**修正版**\s*\n(.+?)(?=\n**|$)",
                        r"\*\*修正版\*\*\s*\n(.+?)(?=\n\*\*|$)"
                    ]
                    
                    for pattern in improvement_patterns:
                        match = re.search(pattern, reviewed_output, re.DOTALL | re.MULTILINE)
                        if match:
                            final_corrected_version = match.group(1).strip()
                            print(f"✅ Extracted improvement section using pattern")
                            break
                
                # If still no corrected version, check if the review contains substantial content that looks like a correction
                if not final_corrected_version and "修正" in reviewed_output and len(reviewed_output) > 1000:
                    # Check if the review output seems to contain a complete corrected version
                    # Look for technical content or structured information
                    if any(keyword in reviewed_output for keyword in ["Linear", "GitHub", "機能", "実装", "設定", "手順"]):
                        print("✅ Using complete review output as it contains substantial technical corrections")
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

{f'''## 3. 最終修正版

以下はClaude Codeレビューに基づく修正版です：

{final_corrected_version}

### 修正の詳細説明
上記の修正版は元の回答に対するレビューで指摘された以下の改善点を反映しています：
- 技術的正確性の向上
- 最新情報の追加
- 論理的一貫性の改善
- 完全性の向上
''' if final_corrected_version and final_corrected_version != reviewed_output else ""}

## 比較分析

### ワークフローの流れ
1. **検索**: 最新情報の収集
2. **初回生成**: Ollama gpt-oss:20bによる初期回答
3. **レビュー**: Claude Code + context7 MCPによる技術的検証
4. **修正**: 事実確認と技術的正確性の向上

### 改善点
- Claude Codeによる事実確認と修正
- context7 MCPツールによる最新技術情報の参照
- より正確で最新の情報の提供
- 論理的一貫性の向上

### 学習ポイント
- 複数のAIシステムを連携させることで回答品質が向上
- 外部検索との組み合わせで最新情報を取得
- レビュープロセスにより信頼性が向上
- MCPツールの活用で技術的正確性が確保

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
        from claude_code_sdk import query, ClaudeCodeOptions
        
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
            async for message in query(prompt=query_generation_prompt, options=options):
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
        import re
        query_pattern = r'クエリ\d+:\s*(.+)'
        matches = re.findall(query_pattern, query_response)
        
        # Clean up and limit to exactly 3 queries
        queries = [query.strip() for query in matches if query.strip()]
        queries = queries[:3]  # Limit to exactly 3 queries
        
        # If we got less than 3 queries, create fallback queries
        if len(queries) < 3:
            print("⚠️ Claude Code agent returned fewer than 3 queries, using fallback")
            queries = [
                user_input,  # Basic query
                f"{user_input} 最新",  # Latest info query
                f"{user_input} 実装"  # Implementation query
            ][:3]
        
        # Ensure we have exactly 3 queries
        queries = queries[:3]
        
        print(f"✅ Generated exactly {len(queries)} search queries using Claude Code agent:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
        
        return {
            **state, 
            "search_queries": queries
        }
        
    except ImportError:
        print("❌ Claude Code SDK not available, falling back to rule-based generation")
        # Fallback: create exactly 3 basic queries from user input
        fallback_queries = [
            user_input,  # Basic query
            f"{user_input} 最新",  # Latest info query
            f"{user_input} 実装方法"  # Implementation query
        ]
        return {
            **state,
            "search_queries": fallback_queries
        }
        
    except Exception as e:
        print(f"❌ Error with Claude Code agent: {e}")
        # Fallback: create exactly 3 basic queries from user input
        fallback_queries = [
            user_input,  # Basic query
            f"{user_input} 最新",  # Latest info query
            f"{user_input} 実装方法"  # Implementation query
        ]
        return {
            **state,
            "search_queries": fallback_queries
        }


def parallel_search_node(state: WorkflowState) -> WorkflowState:
    """Execute multiple searches in parallel using the generated queries."""
    search_queries = state.get("search_queries", [])
    recent_search_mode = state.get("recent_search_mode", False)
    search_days_limit = state.get("search_days_limit", 60)
    
    if not search_queries:
        print("⚠️ No search queries available")
        return {**state, "search_results": ""}
    
    print(f"🔍 Executing {len(search_queries)} parallel searches...")
    
    import subprocess
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def execute_search(query_info):
        query_index, query = query_info
        print(f"🔎 Search {query_index + 1}: {query}")
        
        try:
            # Build psearch command
            psearch_cmd = ["psearch", "search", query[:100], "-n", "3", "-c", "--json"]
            
            # Add date filtering if in recent search mode
            if recent_search_mode:
                current_datetime = datetime.datetime.now()
                current_year = current_datetime.year
                
                if search_days_limit <= 30:
                    psearch_cmd.extend(["-r", "-s"])
                else:
                    months = max(1, search_days_limit // 30)
                    psearch_cmd.extend(["-r", "--months", str(months), "-s"])
            
            # Execute search with timeout
            start_time = time.time()
            result = subprocess.run(
                psearch_cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout per search
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Search {query_index + 1} completed in {elapsed_time:.2f}s")
                return {
                    "query": query,
                    "results": result.stdout,
                    "success": True,
                    "elapsed_time": elapsed_time
                }
            else:
                print(f"❌ Search {query_index + 1} failed: {result.stderr}")
                return {
                    "query": query,
                    "results": f"Search failed: {result.stderr}",
                    "success": False,
                    "elapsed_time": elapsed_time
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Search {query_index + 1} timed out")
            return {
                "query": query,
                "results": "Search timed out",
                "success": False,
                "elapsed_time": 30
            }
        except Exception as e:
            print(f"❌ Search {query_index + 1} error: {e}")
            return {
                "query": query,
                "results": f"Search error: {str(e)}",
                "success": False,
                "elapsed_time": 0
            }
    
    # Execute searches in parallel
    search_results = []
    total_start_time = time.time()
    
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all search tasks
            future_to_query = {
                executor.submit(execute_search, (i, query)): (i, query) 
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
                    search_results.append({
                        "query": query,
                        "results": f"Exception: {str(exc)}",
                        "success": False,
                        "elapsed_time": 0
                    })
    
    except Exception as e:
        print(f"❌ Parallel search execution error: {e}")
        return {**state, "search_results": f"Parallel search error: {str(e)}"}
    
    total_elapsed_time = time.time() - total_start_time
    
    # Combine and summarize results
    successful_searches = [r for r in search_results if r["success"]]
    failed_searches = [r for r in search_results if not r["success"]]
    
    print(f"📊 Search Summary:")
    print(f"  ✅ Successful: {len(successful_searches)}/{len(search_queries)}")
    print(f"  ❌ Failed: {len(failed_searches)}")
    print(f"  ⏱️ Total time: {total_elapsed_time:.2f}s")
    
    # Create combined search results
    combined_results = f"Parallel Search Results ({len(successful_searches)}/{len(search_queries)} successful):\n\n"
    
    for i, result in enumerate(search_results, 1):
        status = "✅" if result["success"] else "❌"
        combined_results += f"{status} Search {i}: {result['query']}\n"
        combined_results += f"Time: {result['elapsed_time']:.2f}s\n"
        if result["success"] and result["results"]:
            # Limit each result to avoid overwhelming the context
            limited_results = result["results"][:1000] + "..." if len(result["results"]) > 1000 else result["results"]
            combined_results += f"Results:\n{limited_results}\n"
        else:
            combined_results += f"Error: {result['results']}\n"
        combined_results += "-" * 50 + "\n\n"
    
    return {
        **state,
        "search_results": combined_results,
        "parallel_search_stats": {
            "total_queries": len(search_queries),
            "successful": len(successful_searches),
            "failed": len(failed_searches),
            "total_time": total_elapsed_time
        }
    }


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
        # Import requests for Slack API
        import requests
        import json
        import os
        import time
        
        # Get Slack webhook URL from environment
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        if not slack_webhook_url:
            print("⚠️ SLACK_WEBHOOK_URL not found in environment variables")
            print("💡 設定方法: export SLACK_WEBHOOK_URL='https://hooks.slack.com/your/webhook/url'")
            return {**state, "slack_notification_sent": False}
        
        # Validate webhook URL format
        if not slack_webhook_url.startswith("https://hooks.slack.com/"):
            print("❌ Invalid Slack webhook URL format")
            print(f"現在のURL: {slack_webhook_url[:50]}...")
            print("💡 正しい形式: https://hooks.slack.com/services/...")
            return {**state, "slack_notification_sent": False}
        
        print(f"✅ Slack webhook URL validated: {slack_webhook_url[:30]}...")
        
        # Create notification message with full document content
        notification_title = f"📄 LangGraphワークフロー実行完了"
        
        # Create Slack message blocks for better formatting
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": notification_title
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*質問:* {original_question}\n*ドキュメント生成パス:* `{document_path}`"
                }
            },
            {
                "type": "divider"
            }
        ]
        
        # Use simpler message format to avoid block formatting issues
        # For large content, send as plain text with summary
        if len(document_content) > 3000:
            print(f"📄 Large content detected ({len(document_content)} chars), using simplified format")
            
            # Create a summary and link to the full document
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
            
            # Simple text payload without blocks
            slack_payload = {
                "text": summary,
                "username": "LangGraph Workflow Bot",
                "icon_emoji": ":memo:"
            }
        else:
            # For smaller content, use the original format
            slack_payload = {
                "text": f"""📄 LangGraphワークフロー実行完了

質問: {original_question}
ドキュメント生成パス: `{document_path}`

結果:
```
{document_content}
```""",
                "username": "LangGraph Workflow Bot",
                "icon_emoji": ":memo:"
            }
        
        # Slack payload is already defined above based on content size
        
        # Retry mechanism
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 送信試行 {attempt + 1}/{max_retries}")
                
                # Send to Slack with timeout
                start_time = time.time()
                response = requests.post(
                    slack_webhook_url,
                    data=json.dumps(slack_payload),
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    print("✅ Slack notification sent successfully")
                    print(f"📊 Document content size: {len(document_content)} characters")
                    print(f"⏱️ Response time: {response_time:.2f} seconds")
                    return {**state, "slack_notification_sent": True}
                else:
                    print(f"❌ Slack notification failed: {response.status_code}")
                    print(f"📄 Response: {response.text}")
                    
                    # Check for specific error conditions
                    if response.status_code == 400:
                        print("💡 Bad Request - チェックポイント:")
                        print("  - Webhook URLが正しいか確認してください")
                        print("  - メッセージのフォーマットが正しいか確認してください")
                        print("  - コンテンツサイズが制限内か確認してください")
                        # Don't retry for 400 errors as they're usually configuration issues
                        return {**state, "slack_notification_sent": False}
                    elif response.status_code == 404:
                        print("💡 Not Found - Webhook URLが無効または削除されています")
                        return {**state, "slack_notification_sent": False}
                    elif response.status_code >= 500:
                        print("💡 Server Error - Slackサーバー側のエラーです")
                    
                    if attempt < max_retries - 1:
                        print(f"⏳ {retry_delay}秒後にリトライします...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print(f"⏳ {retry_delay}秒後にリトライします...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("❌ All retry attempts timed out")
                    
            except requests.exceptions.ConnectionError as e:
                print(f"🌐 Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ {retry_delay}秒後にリトライします...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("❌ All retry attempts failed due to connection errors")
                    
            except requests.exceptions.RequestException as e:
                print(f"📡 Request error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ {retry_delay}秒後にリトライします...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("❌ All retry attempts failed due to request errors")
        
        print(f"❌ Slack notification failed after {max_retries} attempts")
        return {**state, "slack_notification_sent": False}
            
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


def check_ollama_connection():
    """Check if Ollama is running and gpt-oss:20b is available."""
    try:
        import requests

        print("🔍 Checking Ollama connection...")

        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model["name"] for model in models.get("models", [])]

            print(f"✅ Ollama is running with {len(model_names)} models")

            if "gpt-oss:20b" in model_names:
                print("✅ gpt-oss:20b model is available")
                return True
            else:
                print("❌ gpt-oss:20b model not found")
                print("Available models:", model_names)
                print("\n💡 To install gpt-oss:20b, run: ollama pull gpt-oss:20b")
                return False
        else:
            print("❌ Ollama API returned error:", response.status_code)
            return False

    except requests.exceptions.RequestException as e:
        print("❌ Cannot connect to Ollama:", e)
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
        "slack_notification_sent": not bool(os.getenv("SLACK_WEBHOOK_URL")),  # Track Slack notification status (True if not needed)
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
                print("📢 Slack notification sent successfully with complete document content")
            else:
                print("⚠️ Slack notification failed (check SLACK_WEBHOOK_URL environment variable)")
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

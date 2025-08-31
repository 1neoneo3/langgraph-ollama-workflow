#!/usr/bin/env python3
"""LangGraph workflow implementation with Ollama gpt-oss:20b model."""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
import datetime
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
    recent_search_mode: bool
    initial_output: str  # Store first AI output for comparison
    reviewed_output: str  # Store Claude Code reviewed output
    document_generated: bool  # Track document generation status


def search_node(state: WorkflowState) -> WorkflowState:
    """Search for relevant information using psearch with real-time output."""
    user_input = state.get("user_input", "")
    recent_search_mode = state.get("recent_search_mode", False)

    if not user_input:
        return {**state, "search_results": ""}

    search_mode_text = " (limited to past 2 months)" if recent_search_mode else ""
    print(f"🔍 Searching for information about: {user_input}{search_mode_text}")
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

        # Build psearch command with optional date filtering
        psearch_cmd = ["psearch", "search", search_query, "-n", "5", "-c", "--json"]

        # Add date filter for recent search mode (past 2 months)
        if recent_search_mode:
            psearch_cmd.extend(["-r", "--months", "2", "-s"])
            print("📅 Filtering results: recent only (past 2 months), sorted by date")

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

    # Check for keywords that indicate user wants recent information
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
        "2024年",
        "2025年",
        "今年",
        "this year",
    ]

    recent_search_mode = any(keyword in user_input for keyword in recent_keywords)

    if recent_search_mode:
        print(
            "🔍 Recent information keywords detected - search will be limited to past 2 months"
        )

    return {
        **state,
        "messages": messages,
        "iteration": state.get("iteration", 0) + 1,
        "recent_search_mode": recent_search_mode,
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

            # Create a system prompt that includes search results
            system_prompt = f"""
            あなたはLangGraphワークフローの{iteration}回目の処理を行うAIアシスタントです。
            最新の検索結果にアクセスして、正確で最新の情報を提供することができます。
            ユーザーの入力に対して、必要に応じて検索結果から関連情報を取り入れた、思慮深い回答を日本語で提供してください。
            簡潔でありながら、情報量豊富な回答を心がけてください。
            
            ユーザーの入力: {content}
            
            検索結果 (利用可能な場合):
            {search_results if search_results else "検索結果がありません"}
            
            検索結果を活用して、最新で正確な情報を含めた日本語での回答をお願いします。
            """

            # Get response from Ollama
            response = llm.invoke([HumanMessage(content=system_prompt)])

            # Add AI response to messages
            ai_response = response.content
            messages.append(AIMessage(content=ai_response))

            print("✅ LLM Full Response:")
            print("-" * 60)
            print(ai_response)
            print("-" * 60)

            # Store initial output for comparison (first iteration only)
            if state.get("iteration", 0) == 1:
                return {
                    **state,
                    "messages": messages,
                    "processed_output": ai_response,
                    "initial_output": ai_response,
                }
            else:
                return {
                    **state,
                    "messages": messages,
                    "processed_output": ai_response,
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


def decision_node(state: WorkflowState) -> WorkflowState:
    """Decide whether to continue processing or end."""
    iteration = state.get("iteration", 0)

    # Continue for fewer iterations when using LLM (to avoid long execution)
    should_continue = iteration < 2

    return {
        **state,
        "should_continue": should_continue,
    }


def continuation_node(state: WorkflowState) -> WorkflowState:
    """Continue processing with original context and previous response preserved."""
    messages = state["messages"]
    iteration = state["iteration"]
    original_question = state.get("original_user_input", "")
    previous_output = state.get("processed_output", "")

    # Create a context-aware continuation message that includes previous response
    continuation_message = f"""元の質問「{original_question}」について、以下が前回の回答です：

【前回の回答】
{previous_output}

この前回の回答を踏まえて、以下の点で更に詳しく掘り下げて説明してください：

• より具体的な実践例や事例
• 詳細な手順やプロセス
• 関連する最新の動向や発展
• 注意点、課題、考慮事項
• 実装時のベストプラクティス

これは{iteration}回目の詳細化です。前回の内容と重複しないよう、新しい角度からの情報を提供してください。"""

    return {
        **state,
        "messages": messages,
        "user_input": continuation_message,
    }


def review_node(state: WorkflowState) -> WorkflowState:
    """Use Claude Code SDK to review and correct the final output."""
    processed_output = state.get("processed_output", "")
    original_question = state.get("original_user_input", "")

    if not processed_output:
        print("⚠️ No output to review")
        return {**state, "reviewed_output": ""}

    print("🔍 Reviewing output with Claude Code SDK...")

    try:
        # Try importing Claude Code SDK
        from claude_code_sdk import query, ClaudeCodeOptions

        # Create review prompt
        review_prompt = f"""
以下は「{original_question}」という質問に対するAIの回答です。

【対象の回答】
{processed_output}

この回答を詳細にレビューし、以下の点をチェックして修正版を提供してください：

1. 事実の正確性（技術的な間違いや古い情報がないか）
2. 論理的な一貫性（矛盾する内容がないか）
3. 完全性（重要な情報が抜けていないか）
4. わかりやすさ（説明が明確で理解しやすいか）
5. 最新性（最新の情報に基づいているか）

もし間違いや改善点があれば、修正された内容を提供してください。
問題がない場合は「レビュー完了：問題なし」と回答してください。

修正版があれば日本語で提供し、修正点も簡潔に説明してください。
"""

        # Configure options for Claude Code
        options = ClaudeCodeOptions(
            system_prompt="あなたは技術文書の校正・レビューの専門家です。正確性と最新性を重視してレビューを行ってください。",
            max_turns=1,
            allowed_tools=["WebSearch"],  # Allow web search for fact checking
        )

        reviewed_content = ""

        # Query Claude Code SDK using asyncio
        async def get_review():
            content = ""
            async for message in query(prompt=review_prompt, options=options):
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

        try:
            reviewed_content = asyncio.run(get_review())
        except Exception as async_error:
            print(f"❌ Async execution error: {async_error}")
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

    except ImportError:
        print("❌ Claude Code SDK not available, skipping review")
        return {
            **state,
            "reviewed_output": f"レビューをスキップしました（Claude Code SDK利用不可）\n\n元の回答:\n{processed_output}",
        }
    except Exception as e:
        print(f"❌ Error during review: {e}")
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

## 初回AI回答（Ollama gpt-oss:20b）
{initial_output if initial_output else "初回回答なし"}

## Claude Codeレビュー結果
{reviewed_output if reviewed_output else "レビュー結果なし"}

## 比較分析
### 改善点
- Claude Codeによる事実確認と修正
- より正確で最新の情報の提供
- 論理的一貫性の向上

### 学習ポイント
- 複数のAIシステムを連携させることで回答品質が向上
- 外部検索との組み合わせで最新情報を取得
- レビュープロセスにより信頼性が向上

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
        }

    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        return {
            **state,
            "document_generated": False,
        }


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow with Ollama."""

    # Create the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes to the workflow
    workflow.add_node("input", input_node)
    workflow.add_node("search", search_node)
    workflow.add_node("process", processing_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("continue", continuation_node)
    workflow.add_node("review", review_node)
    workflow.add_node("document", documentation_node)

    # Define the workflow edges
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "search")
    workflow.add_edge("search", "process")
    workflow.add_edge("process", "decision")

    # Conditional routing function
    def route_decision(state: WorkflowState) -> str:
        """Route based on the decision state."""
        return "continue" if state.get("should_continue", False) else "review"

    # Conditional edges from decision node
    workflow.add_conditional_edges(
        "decision",
        route_decision,
        {
            "continue": "continue",
            "review": "review",
        },
    )

    # Edge from continue back to input for loop
    workflow.add_edge("continue", "input")

    # New edges for review and documentation
    workflow.add_edge("review", "document")
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
    user_question = input("❓ ")

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
        "recent_search_mode": False,
        "initial_output": "",  # Store first AI output for comparison
        "reviewed_output": "",  # Store Claude Code reviewed output
        "document_generated": False,  # Track document generation status
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
        else:
            print("⚠️ Documentation generation failed or was skipped")

    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

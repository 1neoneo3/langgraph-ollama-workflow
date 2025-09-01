"""Documentation service for workflow result generation."""

import datetime
import re
from pathlib import Path
from typing import Dict

from ..core.state import WorkflowState


def create_document_filename(original_question: str) -> str:
    """Create a safe filename from the original question."""
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
    
    return f"{question_summary}_分析結果.md"


def extract_corrected_version(reviewed_output: str) -> str:
    """Extract the corrected version from review output using various patterns."""
    if not reviewed_output:
        return ""

    # Look for patterns like "修正版:" or actual corrected text sections
    corrected_patterns = [
        r"修正版[：:]\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
        r"修正[：:]\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
        r"改善版[：:]\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
        r"以下が修正版です[：:]?\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
        r"修正後[：:]?\s*\n(.+?)(?=\n\n##|\n\n---|\Z)",
    ]

    for pattern in corrected_patterns:
        match = re.search(pattern, reviewed_output, re.DOTALL | re.MULTILINE)
        if match:
            final_corrected_version = match.group(1).strip()
            print(f"✅ Extracted corrected version using pattern: {pattern[:20]}...")
            return final_corrected_version

    # If no explicit corrected version found, check for structured corrections
    improvement_patterns = [
        r"## レビュー結果.*?## 修正内容.*?\n(.+?)(?=\n## |$)",
        r"### 修正内容\s*\n(.+?)(?=\n### |$)",
        r"**修正版**\s*\n(.+?)(?=\n**|$)",
        r"\*\*修正版\*\*\s*\n(.+?)(?=\n\*\*|$)",
    ]

    for pattern in improvement_patterns:
        match = re.search(pattern, reviewed_output, re.DOTALL | re.MULTILINE)
        if match:
            final_corrected_version = match.group(1).strip()
            print("✅ Extracted improvement section using pattern")
            return final_corrected_version

    # If still no corrected version, check if the review contains substantial content
    if (
        "修正" in reviewed_output
        and len(reviewed_output) > 1000
        and any(
            keyword in reviewed_output
            for keyword in ["Linear", "GitHub", "機能", "実装", "設定", "手順"]
        )
    ):
        print("✅ Using complete review output as it contains substantial technical corrections")
        return reviewed_output

    return ""


def generate_markdown_content(state: WorkflowState, final_corrected_version: str) -> str:
    """Generate the markdown content for documentation."""
    original_question = state.get("original_user_input", "")
    initial_output = state.get("initial_output", "")
    reviewed_output = state.get("reviewed_output", "")
    search_results = state.get("search_results", "")

    corrected_section = ""
    if final_corrected_version and final_corrected_version != reviewed_output:
        corrected_section = f'''## 3. 最終修正版

以下はClaude Codeレビューに基づく修正版です：

{final_corrected_version}

### 修正の詳細説明
上記の修正版は元の回答に対するレビューで指摘された以下の改善点を反映しています：
- 技術的正確性の向上
- 最新情報の追加
- 論理的一貫性の改善
- 完全性の向上
'''

    return f"""# LangGraphワークフロー実行結果

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

{corrected_section}

---
*このドキュメントは LangGraph + Claude Code SDK ワークフローにより自動生成されました*
"""


def documentation_node(state: WorkflowState) -> WorkflowState:
    """Generate markdown documentation comparing initial and final outputs."""
    original_question = state.get("original_user_input", "")
    reviewed_output = state.get("reviewed_output", "")

    print("📝 Generating documentation...")

    try:
        # Create docs directory if it doesn't exist
        docs_dir = Path.home() / "workspace" / "Docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Create filename and path
        filename = create_document_filename(original_question)
        file_path = docs_dir / filename

        # Extract final corrected version if available
        final_corrected_version = extract_corrected_version(reviewed_output)

        # Generate markdown content
        markdown_content = generate_markdown_content(state, final_corrected_version)

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
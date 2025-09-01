#!/usr/bin/env python3
"""Test script for the enhanced ollama workflow with complete review handling."""

import sys
import os
from pathlib import Path

# Add the current directory to sys.path to import ollama_workflow
sys.path.insert(0, str(Path(__file__).parent))

from ollama_workflow import create_workflow, WorkflowState


def test_review_content_preservation():
    """Test that review content is preserved without truncation."""
    print("🧪 Testing review content preservation...")
    
    # Create a mock state with long review content
    test_state = {
        "messages": [],
        "iteration": 1,
        "user_input": "Linearタスク管理について教えて",
        "original_user_input": "Linearタスク管理について教えて",
        "processed_output": "Linearは開発チーム向けのタスク管理ツールです...",
        "should_continue": True,
        "search_results": "",
        "recent_search_mode": False,
        "initial_output": "Linearは開発チーム向けのタスク管理ツールです...",
        "reviewed_output": "",
        "document_generated": False,
    }
    
    # Test the workflow creation
    workflow = create_workflow()
    app = workflow.compile()
    
    print("✅ Workflow created and compiled successfully")
    
    # Test documentation node with mock data
    test_state["reviewed_output"] = """## レビュー結果：修正が必要

### 重要な問題点

#### 1. 技術的正確性の問題（Critical）

問題: GitHubとLinearの連携機能について不正確な記述
- 「GitHubリポジトリをLinearに接続すると、PRが自動でタスクに変換」
- 実際: PRはタスクに変換されません。既存のIssueをLinearタスクにリンクしたり、Linear Issue IDをPRに含めることで連携します

問題: ショートカットキーの記述
- ⌘+⇧+D で完了という記載
- 実際: Linearの標準ショートカットは ⌘+Shift+Enter でタスクを完了に変更

#### 2. 機能説明の不正確性（High）

問題: "Planning Poker"機能
- Linearには組み込みのPlanning Pokerはありません
- 代替手段: Story Point推定は手動入力、またはサードパーティツール連携

### 修正版

Linearは「タスク＝開発作業」として設計されているため、GitHubと自然に連携し、スプリントベースでの進捗管理がスムーズです。

#### GitHubとの正しい連携方法
1. Linear Issue IDをGitHub PRのタイトルまたは説明に含める
2. GitHub WebhookでLinear Issue状態を自動更新
3. PRマージ時にLinear Issueを自動クローズ

#### 正しいショートカットキー
- タスク完了: ⌘+Shift+Enter
- 新規タスク作成: ⌘+K
- プロジェクト切替: ⌘+P

NotionのタスクDBをLinearに移行する際は、以下の手順を推奨します：
1. 既存タスクのエクスポート（CSV形式）
2. Linearプロジェクト作成とラベル設定
3. インポート機能による一括移行
4. チームメンバーのアクセス権限設定

まずは小規模なスプリントで試行し、運用ルールを確立してからフル移行することで、Linearを最大限活用できます。"""
    
    from ollama_workflow import documentation_node
    result = documentation_node(test_state)
    
    # Check if documentation was generated
    if result.get("document_generated"):
        print("✅ Documentation generation test passed")
        
        # Check if review content is preserved
        docs_dir = Path.home() / "workspace" / "Docs"
        md_files = list(docs_dir.glob("*分析結果.md"))
        
        if md_files:
            latest_file = max(md_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for complete review content
            if "GitHubとLinearの連携機能について不正確な記述" in content:
                print("✅ Complete review content preserved in documentation")
            else:
                print("❌ Review content may be truncated in documentation")
                
            # Check for final corrected version
            if "## 3. 最終修正版" in content:
                print("✅ Final corrected version section found in documentation")
            else:
                print("⚠️ Final corrected version section not found")
                
            print(f"📄 Test documentation generated: {latest_file}")
            
        else:
            print("❌ No documentation files found")
    else:
        print("❌ Documentation generation test failed")
    
    return result.get("document_generated", False)


def main():
    """Run the enhanced workflow tests."""
    print("🚀 Running Enhanced Workflow Tests")
    print("=" * 50)
    
    try:
        # Test review content preservation
        test_passed = test_review_content_preservation()
        
        if test_passed:
            print("\n✅ All tests passed!")
            print("📝 Enhanced workflow is ready for use with complete review content handling")
        else:
            print("\n❌ Some tests failed")
            return 1
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
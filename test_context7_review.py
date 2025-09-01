#!/usr/bin/env python3
"""Test context7 MCP functionality from review_node specifically"""

import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

def test_review_with_context7():
    """Test the exact context7 configuration used in review_node"""
    
    # Test data mimicking the review_node scenario
    processed_output = "Reactでは、useState()フックを使って状態管理ができます。const [state, setState] = useState(initialValue)の形で使用します。"
    original_question = "ReactのuseStateフックの使い方を教えて"
    
    # Create exact review prompt from review_node
    review_prompt = f"""
以下は「{original_question}」という質問に対するAIの回答です。

【対象の回答】
{processed_output}

まず、元の質問がプログラミング、開発、技術的な内容に関連しているかを判定してください。
プログラミング・技術関連の場合は、context7 MCPツールを使用して公式ドキュメントや最新の技術情報を参照し、より詳細なレビューを行ってください。

この回答を詳細にレビューし、以下の点をチェックして修正版を提供してください：

**一般的なレビューポイント：**
1. 事実の正確性（間違いや古い情報がないか）
2. 論理的な一貫性（矛盾する内容がないか）
3. 完全性（重要な情報が抜けていないか）
4. わかりやすさ（説明が明確で理解しやすいか）
5. 最新性（最新の情報に基づいているか）

**技術的な質問の場合の追加レビューポイント：**
- 技術的正確性（コードの構文、APIの使用方法）
- ベストプラクティス準拠（業界標準に従っているか）
- セキュリティ（セキュリティ上の問題やリスクがないか）
- パフォーマンス（効率的で最適化されたアプローチか）
- 公式ドキュメントとの整合性
- 実装時の注意点や落とし穴

技術的な質問の場合は、context7 MCPツールを活用して公式ドキュメントを確認し、最新で正確な技術情報に基づいてレビューしてください。

修正版があれば日本語で提供し、修正点と理由を詳細に説明してください。
問題がない場合は「レビュー完了：問題なし」と回答してください。
"""
    
    # Configure exact options from review_node
    options = ClaudeCodeOptions(
        system_prompt="あなたは技術文書の校正・レビューの専門家です。質問が技術的な内容の場合は、context7 MCPツールを積極的に使用して公式ドキュメントや最新の技術情報を参照し、正確性と実用性を重視してレビューを行ってください。",
        max_turns=1,
        allowed_tools=["WebSearch"],
        mcp_servers={
            "context7": {
                "command": "npx",
                "args": ["-y", "@context7/server"]
            }
        }
    )
    
    print("🧪 Testing context7 MCP in review scenario...")
    print("📝 Input:")
    print(f"  Question: {original_question}")
    print(f"  Original output: {processed_output}")
    print()
    
    async def get_review():
        content = ""
        try:
            async for message in query(prompt=review_prompt, options=options):
                if hasattr(message, "content"):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if hasattr(block, "text"):
                                content += block.text
                                print(".", end="", flush=True)
                    else:
                        content += str(message.content)
                        print(".", end="", flush=True)
            return content
        except Exception as e:
            return f"Error: {e}"
    
    try:
        reviewed_content = asyncio.run(get_review())
        
        print("\n✅ Review completed!")
        print("📄 Results:")
        print(f"  Response length: {len(reviewed_content)}")
        print()
        print("📋 Review output:")
        print("-" * 60)
        print(reviewed_content)
        print("-" * 60)
        
        # Analysis
        success_indicators = [
            "context7" in reviewed_content.lower(),
            "公式ドキュメント" in reviewed_content,
            len(reviewed_content) > len(processed_output) * 2,
            "React" in reviewed_content and "useState" in reviewed_content
        ]
        
        success_count = sum(success_indicators)
        print(f"\n📊 Success indicators: {success_count}/4")
        
        if success_count >= 2:
            print("✅ Context7 MCP integration appears to be working")
        else:
            print("⚠️ Context7 MCP integration may have issues")
            
        return success_count >= 2
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Context7 MCP Review Test")
    print("=" * 60)
    
    success = test_review_with_context7()
    
    print("\n" + "=" * 60)
    print(f"Final Result: {'✅ WORKING' if success else '❌ ISSUES DETECTED'}")

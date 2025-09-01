#!/usr/bin/env python3
"""
Slack統合テスト用のスクリプト
SLACK_WEBHOOK_URL環境変数の検証とSlack通知機能のテスト
"""

import os
import requests
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()


@dataclass
class SlackTestResult:
    """Slackテスト結果を格納するデータクラス"""
    test_name: str
    success: bool
    message: str
    response_code: Optional[int] = None
    response_time: Optional[float] = None
    error_details: Optional[str] = None


class SlackIntegrationTester:
    """Slack統合テスト用のクラス"""
    
    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.test_results: List[SlackTestResult] = []
    
    def validate_webhook_url(self) -> SlackTestResult:
        """SLACK_WEBHOOK_URL環境変数の検証"""
        print("🔍 環境変数SLACK_WEBHOOK_URLの検証中...")
        
        if not self.webhook_url:
            return SlackTestResult(
                test_name="環境変数検証",
                success=False,
                message="SLACK_WEBHOOK_URL環境変数が設定されていません",
                error_details="環境変数を設定してください: export SLACK_WEBHOOK_URL='your_webhook_url'"
            )
        
        if not self.webhook_url.startswith("https://hooks.slack.com/"):
            return SlackTestResult(
                test_name="環境変数検証",
                success=False,
                message="無効なSlack Webhook URL形式",
                error_details=f"URL: {self.webhook_url[:50]}... (無効な形式)"
            )
        
        return SlackTestResult(
            test_name="環境変数検証",
            success=True,
            message="SLACK_WEBHOOK_URL環境変数が正常に設定されています"
        )
    
    def test_basic_notification(self) -> SlackTestResult:
        """基本的なSlack通知テスト"""
        print("📤 基本的なSlack通知テスト中...")
        
        if not self.webhook_url:
            return SlackTestResult(
                test_name="基本通知テスト",
                success=False,
                message="Webhook URLが設定されていません"
            )
        
        test_message = {
            "text": "🧪 LangGraph Workflow Slack統合テスト",
            "username": "LangGraph Test Bot",
            "icon_emoji": ":test_tube:",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*テスト実行時刻:* " + time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.webhook_url,
                data=json.dumps(test_message),
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return SlackTestResult(
                    test_name="基本通知テスト",
                    success=True,
                    message="基本的なSlack通知が正常に送信されました",
                    response_code=response.status_code,
                    response_time=response_time
                )
            else:
                return SlackTestResult(
                    test_name="基本通知テスト",
                    success=False,
                    message=f"Slack通知が失敗しました (Status: {response.status_code})",
                    response_code=response.status_code,
                    response_time=response_time,
                    error_details=response.text
                )
                
        except requests.exceptions.Timeout:
            return SlackTestResult(
                test_name="基本通知テスト",
                success=False,
                message="リクエストがタイムアウトしました (30秒)",
                error_details="ネットワーク接続またはSlackサーバーの応答が遅い可能性があります"
            )
        except requests.exceptions.RequestException as e:
            return SlackTestResult(
                test_name="基本通知テスト",
                success=False,
                message="ネットワークエラーが発生しました",
                error_details=str(e)
            )
        except Exception as e:
            return SlackTestResult(
                test_name="基本通知テスト",
                success=False,
                message="予期しないエラーが発生しました",
                error_details=str(e)
            )
    
    def test_large_content_notification(self) -> SlackTestResult:
        """大きなコンテンツでのSlack通知テスト"""
        print("📄 大きなコンテンツでのSlack通知テスト中...")
        
        if not self.webhook_url:
            return SlackTestResult(
                test_name="大きなコンテンツテスト",
                success=False,
                message="Webhook URLが設定されていません"
            )
        
        # 実際のワークフロー出力をシミュレート
        large_content = """# LangGraphワークフロー実行結果

## 実行情報
- **実行日時**: 2024-01-15 10:30:45
- **質問**: Linear Issueの最新機能について教えて
- **ワークフローイテレーション**: 1

## 元の質問
```
Linear Issueの最新機能について教えて
```

## 検索結果の概要
```
Linear Issue management platform の最新機能に関する検索結果...
多数の新機能とアップデートが見つかりました...
```

## 1. 初回AI回答（Ollama gpt-oss:20b）
Linear Issueは最新のプロジェクト管理ツールです。
2024年の主要な新機能には以下があります：

1. **AIアシスタント機能**
   - 自動的なイシューの分類
   - 優先度の自動設定
   - 関連イシューの提案

2. **高度なフィルタリング**
   - カスタムビューの作成
   - 動的フィルター
   - 保存されたクエリ

3. **統合機能の強化**
   - GitHub統合の改善
   - Slack連携の強化
   - API v2.0の提供

## 2. Claude Codeレビュー・修正結果
レビュー完了：最新情報を含む正確な回答です。
技術的な詳細と実装例も適切に含まれています。

## 3. 最終修正版

Linear Issueプラットフォームの2024年最新機能：

### 主要な新機能
1. **AIアシスタント機能**
   - イシューの自動分類とタグ付け
   - 優先度の自動設定
   - 関連イシューとプルリクエストの提案

2. **ワークフロー自動化**
   - カスタムオートメーション
   - 条件付きアクション
   - 外部ツールとの統合

3. **パフォーマンス向上**
   - リアルタイム同期
   - 高速検索機能
   - モバイルアプリの最適化

### 実装の詳細
- API v2.0での新エンドポイント
- GraphQLクエリの最適化
- リアルタイム通知システム

---
*このドキュメントは LangGraph + Claude Code SDK ワークフローにより自動生成されました*
"""
        
        test_message = {
            "text": "🧪 LangGraph Workflow 大きなコンテンツテスト",
            "username": "LangGraph Test Bot",
            "icon_emoji": ":memo:",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "📄 大きなコンテンツテスト"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"コンテンツサイズ: {len(large_content)} 文字"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"```\n{large_content}\n```"
                    }
                }
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.webhook_url,
                data=json.dumps(test_message),
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return SlackTestResult(
                    test_name="大きなコンテンツテスト",
                    success=True,
                    message=f"大きなコンテンツ ({len(large_content)}文字) の送信が成功しました",
                    response_code=response.status_code,
                    response_time=response_time
                )
            else:
                return SlackTestResult(
                    test_name="大きなコンテンツテスト",
                    success=False,
                    message=f"大きなコンテンツの送信が失敗しました (Status: {response.status_code})",
                    response_code=response.status_code,
                    response_time=response_time,
                    error_details=response.text
                )
                
        except Exception as e:
            return SlackTestResult(
                test_name="大きなコンテンツテスト",
                success=False,
                message="大きなコンテンツテスト中にエラーが発生しました",
                error_details=str(e)
            )
    
    def test_retry_mechanism(self) -> SlackTestResult:
        """リトライ機能のテスト"""
        print("🔄 リトライ機能テスト中...")
        
        if not self.webhook_url:
            return SlackTestResult(
                test_name="リトライ機能テスト",
                success=False,
                message="Webhook URLが設定されていません"
            )
        
        max_retries = 3
        retry_delay = 1  # 1秒
        
        test_message = {
            "text": "🔄 リトライ機能テスト",
            "username": "LangGraph Test Bot",
            "icon_emoji": ":repeat:"
        }
        
        for attempt in range(max_retries):
            try:
                print(f"  試行 {attempt + 1}/{max_retries}")
                start_time = time.time()
                
                response = requests.post(
                    self.webhook_url,
                    data=json.dumps(test_message),
                    headers={'Content-Type': 'application/json'},
                    timeout=10  # より短いタイムアウト
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    return SlackTestResult(
                        test_name="リトライ機能テスト",
                        success=True,
                        message=f"リトライ機能テストが成功しました (試行 {attempt + 1}/{max_retries})",
                        response_code=response.status_code,
                        response_time=response_time
                    )
                else:
                    print(f"  試行 {attempt + 1} 失敗: Status {response.status_code}")
                    if attempt < max_retries - 1:
                        print(f"  {retry_delay}秒後にリトライします...")
                        time.sleep(retry_delay)
                        
            except Exception as e:
                print(f"  試行 {attempt + 1} エラー: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"  {retry_delay}秒後にリトライします...")
                    time.sleep(retry_delay)
        
        return SlackTestResult(
            test_name="リトライ機能テスト",
            success=False,
            message=f"リトライ機能テストが失敗しました ({max_retries}回試行)",
            error_details="すべての試行が失敗しました"
        )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """すべてのテストを実行"""
        print("🚀 Slack統合テスト開始")
        print("=" * 60)
        
        # テスト実行
        tests = [
            self.validate_webhook_url,
            self.test_basic_notification,
            self.test_large_content_notification,
            self.test_retry_mechanism
        ]
        
        results = []
        for test_func in tests:
            result = test_func()
            results.append(result)
            self.test_results.append(result)
            
            # 結果の表示
            status_icon = "✅" if result.success else "❌"
            print(f"\n{status_icon} {result.test_name}")
            print(f"   {result.message}")
            
            if result.response_code:
                print(f"   レスポンスコード: {result.response_code}")
            if result.response_time:
                print(f"   応答時間: {result.response_time:.2f}秒")
            if result.error_details:
                print(f"   エラー詳細: {result.error_details}")
            
            # テスト間の間隔
            if test_func != tests[-1]:  # 最後のテストでない場合
                time.sleep(2)
        
        # 結果サマリー
        successful_tests = [r for r in results if r.success]
        failed_tests = [r for r in results if not r.success]
        
        print("\n" + "=" * 60)
        print("📊 テスト結果サマリー")
        print(f"  総テスト数: {len(results)}")
        print(f"  ✅ 成功: {len(successful_tests)}")
        print(f"  ❌ 失敗: {len(failed_tests)}")
        print(f"  成功率: {len(successful_tests)/len(results)*100:.1f}%")
        
        if failed_tests:
            print("\n❌ 失敗したテスト:")
            for test in failed_tests:
                print(f"  - {test.test_name}: {test.message}")
        
        # 推奨事項
        print("\n💡 推奨事項:")
        if not self.webhook_url:
            print("  - SLACK_WEBHOOK_URL環境変数を設定してください")
            print("  - Slack Appの設定でIncoming Webhookを有効にしてください")
        elif len(successful_tests) == len(results):
            print("  - すべてのテストが成功しました！")
            print("  - Slack統合は正常に動作しています")
        elif len(successful_tests) > 0:
            print("  - 一部のテストが成功しています")
            print("  - 失敗したテストのエラー詳細を確認してください")
        
        return {
            "total_tests": len(results),
            "successful": len(successful_tests),
            "failed": len(failed_tests),
            "success_rate": len(successful_tests)/len(results)*100,
            "results": results
        }


def main():
    """メイン実行関数"""
    tester = SlackIntegrationTester()
    test_results = tester.run_all_tests()
    
    # 終了コードの決定
    if test_results["failed"] == 0:
        print("\n🎉 すべてのテストが成功しました！")
        return 0
    else:
        print(f"\n⚠️ {test_results['failed']}個のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit(main())
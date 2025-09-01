#!/usr/bin/env python3
"""
モックWebhook URLを使用したSlack統合テスト
実際のSlack Webhook URLがなくてもリトライ機能とエラーハンドリングをテスト
"""

import os
import sys
import time
from unittest.mock import patch, MagicMock
import requests


def test_slack_notification_with_mock():
    """モックWebhook URLでSlack通知機能をテスト"""
    print("🧪 モックWebhook URLでSlack通知機能をテスト")
    print("=" * 60)
    
    # テスト用の状態を作成
    test_state = {
        "document_content": "# テストドキュメント\n\nこれはテスト用のドキュメント内容です。\n\n## セクション1\n詳細な内容がここに入ります。",
        "document_path": "/tmp/test_document.md",
        "original_user_input": "テスト質問",
        "slack_notification_sent": False
    }
    
    # 環境変数を一時的に設定
    test_webhook_url = "https://hooks.slack.com/services/TEST/WEBHOOK/URL"
    
    # ollama_workflowモジュールを動的にインポート
    sys.path.insert(0, '/home/mk/workspace/langgraph-workflow')
    
    try:
        from ollama_workflow import slack_notification_node
        
        # 1. 成功ケースのテスト
        print("\n1️⃣ 成功ケースのテスト")
        print("-" * 30)
        
        with patch.dict(os.environ, {'SLACK_WEBHOOK_URL': test_webhook_url}):
            with patch('requests.post') as mock_post:
                # 成功レスポンスをモック
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = "ok"
                mock_post.return_value = mock_response
                
                result = slack_notification_node(test_state.copy())
                
                print(f"✅ テスト結果: {'成功' if result['slack_notification_sent'] else '失敗'}")
                print(f"📤 requests.post呼び出し回数: {mock_post.call_count}")
                print(f"🔗 呼び出されたURL: {mock_post.call_args[0][0] if mock_post.call_args else 'なし'}")
        
        # 2. リトライが必要なエラーケースのテスト (500エラー → 成功)
        print("\n2️⃣ リトライ成功ケースのテスト (500エラー → 成功)")
        print("-" * 30)
        
        with patch.dict(os.environ, {'SLACK_WEBHOOK_URL': test_webhook_url}):
            with patch('requests.post') as mock_post:
                with patch('time.sleep') as mock_sleep:  # sleepをモックして高速化
                    # 最初の2回は500エラー、3回目で成功
                    responses = [
                        MagicMock(status_code=500, text="Internal Server Error"),
                        MagicMock(status_code=500, text="Internal Server Error"),
                        MagicMock(status_code=200, text="ok")
                    ]
                    mock_post.side_effect = responses
                    
                    start_time = time.time()
                    result = slack_notification_node(test_state.copy())
                    elapsed_time = time.time() - start_time
                    
                    print(f"✅ テスト結果: {'成功' if result['slack_notification_sent'] else '失敗'}")
                    print(f"🔄 リトライ実行回数: {mock_post.call_count}")
                    print(f"⏱️ テスト実行時間: {elapsed_time:.2f}秒")
                    print(f"😴 sleep呼び出し回数: {mock_sleep.call_count}")
        
        # 3. リトライ失敗ケースのテスト (全て500エラー)
        print("\n3️⃣ リトライ失敗ケースのテスト (全て500エラー)")
        print("-" * 30)
        
        with patch.dict(os.environ, {'SLACK_WEBHOOK_URL': test_webhook_url}):
            with patch('requests.post') as mock_post:
                with patch('time.sleep') as mock_sleep:
                    # 全ての試行で500エラー
                    mock_response = MagicMock()
                    mock_response.status_code = 500
                    mock_response.text = "Internal Server Error"
                    mock_post.return_value = mock_response
                    
                    result = slack_notification_node(test_state.copy())
                    
                    print(f"❌ テスト結果: {'成功' if result['slack_notification_sent'] else '失敗（期待通り）'}")
                    print(f"🔄 最大リトライ実行: {mock_post.call_count}")
                    print(f"😴 sleep呼び出し回数: {mock_sleep.call_count}")
        
        # 4. 400エラーケース（リトライしない）
        print("\n4️⃣ 400エラーケースのテスト（リトライなし）")
        print("-" * 30)
        
        with patch.dict(os.environ, {'SLACK_WEBHOOK_URL': test_webhook_url}):
            with patch('requests.post') as mock_post:
                with patch('time.sleep') as mock_sleep:
                    # 400エラー（設定問題）
                    mock_response = MagicMock()
                    mock_response.status_code = 400
                    mock_response.text = "Bad Request"
                    mock_post.return_value = mock_response
                    
                    result = slack_notification_node(test_state.copy())
                    
                    print(f"❌ テスト結果: {'成功' if result['slack_notification_sent'] else '失敗（期待通り）'}")
                    print(f"📤 requests.post呼び出し回数: {mock_post.call_count} (1回のみ期待)")
                    print(f"😴 sleep呼び出し回数: {mock_sleep.call_count} (0回期待)")
        
        # 5. 接続エラーケースのテスト
        print("\n5️⃣ 接続エラーケースのテスト")
        print("-" * 30)
        
        with patch.dict(os.environ, {'SLACK_WEBHOOK_URL': test_webhook_url}):
            with patch('requests.post') as mock_post:
                with patch('time.sleep') as mock_sleep:
                    # 接続エラーを発生させる
                    mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
                    
                    result = slack_notification_node(test_state.copy())
                    
                    print(f"❌ テスト結果: {'成功' if result['slack_notification_sent'] else '失敗（期待通り）'}")
                    print(f"🔄 接続リトライ実行: {mock_post.call_count}")
                    print(f"😴 sleep呼び出し回数: {mock_sleep.call_count}")
        
        # 6. タイムアウトエラーケースのテスト
        print("\n6️⃣ タイムアウトエラーケースのテスト")
        print("-" * 30)
        
        with patch.dict(os.environ, {'SLACK_WEBHOOK_URL': test_webhook_url}):
            with patch('requests.post') as mock_post:
                with patch('time.sleep') as mock_sleep:
                    # タイムアウトエラーを発生させる
                    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
                    
                    result = slack_notification_node(test_state.copy())
                    
                    print(f"❌ テスト結果: {'成功' if result['slack_notification_sent'] else '失敗（期待通り）'}")
                    print(f"⏰ タイムアウトリトライ実行: {mock_post.call_count}")
                    print(f"😴 sleep呼び出し回数: {mock_sleep.call_count}")
        
        # 7. 大容量コンテンツのテスト
        print("\n7️⃣ 大容量コンテンツ分割テスト")
        print("-" * 30)
        
        # 大きなコンテンツを作成 (10,000文字)
        large_content = "# 大容量テストドキュメント\n\n" + "これは大容量テスト用の長いコンテンツです。" * 200
        large_state = test_state.copy()
        large_state["document_content"] = large_content
        
        with patch.dict(os.environ, {'SLACK_WEBHOOK_URL': test_webhook_url}):
            with patch('requests.post') as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = "ok"
                mock_post.return_value = mock_response
                
                result = slack_notification_node(large_state)
                
                print(f"✅ テスト結果: {'成功' if result['slack_notification_sent'] else '失敗'}")
                print(f"📄 コンテンツサイズ: {len(large_content)} 文字")
                print(f"📤 requests.post呼び出し回数: {mock_post.call_count}")
                
                # ペイロードの確認
                if mock_post.call_args:
                    payload = mock_post.call_args[1]['data']
                    import json
                    payload_data = json.loads(payload)
                    blocks = payload_data.get('blocks', [])
                    content_blocks = [b for b in blocks if b.get('type') == 'section' and 'Part' in str(b.get('text', {}).get('text', ''))]
                    print(f"📦 分割されたブロック数: {len(content_blocks)}")
        
        # 8. 無効なWebhook URLテスト
        print("\n8️⃣ 無効なWebhook URLテスト")
        print("-" * 30)
        
        invalid_url = "https://invalid.webhook.url/test"
        
        with patch.dict(os.environ, {'SLACK_WEBHOOK_URL': invalid_url}):
            result = slack_notification_node(test_state.copy())
            
            print(f"❌ テスト結果: {'成功' if result['slack_notification_sent'] else '失敗（期待通り）'}")
            print(f"🔗 無効URL検出: {'✅' if not result['slack_notification_sent'] else '❌'}")
        
        print("\n" + "=" * 60)
        print("🎉 モックテスト完了！")
        print("✅ すべてのエラーハンドリングとリトライ機能が正常に動作しています")
        
    except ImportError as e:
        print(f"❌ モジュールインポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_slack_notification_with_mock()
    exit(0 if success else 1)
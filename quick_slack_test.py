#!/usr/bin/env python3
"""
簡単なSlack通知テスト
Webhook URL設定後の動作確認用
"""

import os
import requests
import json
import sys
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()


def test_slack_notification():
    """簡単なSlack通知テスト"""
    print("🚀 Slack通知テスト開始")
    print("=" * 40)
    
    # 環境変数の確認
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    if not webhook_url:
        print("❌ SLACK_WEBHOOK_URL環境変数が設定されていません")
        print("\n💡 設定方法:")
        print("export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'")
        return False
    
    print(f"✅ Webhook URL確認: {webhook_url[:30]}...")
    
    # URL形式の確認
    if not webhook_url.startswith("https://hooks.slack.com/"):
        print("❌ 無効なWebhook URL形式です")
        print("正しい形式: https://hooks.slack.com/services/...")
        return False
    
    print("✅ URL形式が正しいです")
    
    # テストメッセージの送信
    test_payload = {
        "text": "🧪 LangGraph Workflow Slack統合テスト",
        "username": "LangGraph Test Bot",
        "icon_emoji": ":test_tube:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🧪 Slack統合テスト成功！"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*テスト実行時刻:* " + __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n*状況:* LangGraph WorkflowからのSlack通知が正常に動作しています！"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "✅ *機能確認項目:*\n• Webhook URL接続\n• メッセージフォーマット\n• ブロック形式の表示\n• 絵文字とユーザー名"
                }
            }
        ]
    }
    
    try:
        print("📤 テストメッセージを送信中...")
        
        response = requests.post(
            webhook_url,
            data=json.dumps(test_payload),
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        if response.status_code == 200:
            print("✅ テストメッセージの送信に成功しました！")
            print("🎉 Slackチャンネルで通知を確認してください")
            print(f"📊 ペイロードサイズ: {len(json.dumps(test_payload))} bytes")
            return True
        else:
            print(f"❌ テストメッセージの送信に失敗しました")
            print(f"ステータスコード: {response.status_code}")
            print(f"エラーメッセージ: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ リクエストがタイムアウトしました（15秒）")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"🌐 接続エラー: {e}")
        print("💡 インターネット接続またはSlackサーバーの状況を確認してください")
        return False
    except requests.exceptions.RequestException as e:
        print(f"📡 リクエストエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False


def main():
    """メイン実行関数"""
    success = test_slack_notification()
    
    if success:
        print("\n🎊 テスト完了！")
        print("これで ollama_workflow.py を実行した際にSlack通知が正常に動作します")
    else:
        print("\n⚠️ テストが失敗しました")
        print("Webhook URLと設定を再確認してください")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
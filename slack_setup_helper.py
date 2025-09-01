#!/usr/bin/env python3
"""
Slack Webhook URL設定ヘルパー
環境変数の設定とテストを支援するスクリプト
"""

import os
import sys
import requests
import json
from typing import Optional


def print_banner():
    """バナーを表示"""
    print("🔧 Slack Webhook URL設定ヘルパー")
    print("=" * 60)


def check_current_setup():
    """現在の設定をチェック"""
    print("📋 現在の設定状況をチェック中...")
    
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    if webhook_url:
        print(f"✅ SLACK_WEBHOOK_URL環境変数が設定されています")
        print(f"   URL: {webhook_url[:30]}...")
        
        # URL形式のバリデーション
        if webhook_url.startswith("https://hooks.slack.com/"):
            print("✅ Webhook URL形式が正しいです")
            return webhook_url
        else:
            print("❌ Webhook URL形式が正しくありません")
            print("   正しい形式: https://hooks.slack.com/services/...")
            return None
    else:
        print("❌ SLACK_WEBHOOK_URL環境変数が設定されていません")
        return None


def provide_setup_instructions():
    """セットアップ手順を表示"""
    print("\n📖 Slack Webhook URL設定手順:")
    print("-" * 40)
    print("1. Slackワークスペースにアクセス")
    print("2. https://api.slack.com/apps にアクセス")
    print("3. 「Create New App」をクリック")
    print("4. 「From scratch」を選択")
    print("5. アプリ名とワークスペースを設定")
    print("6. 左サイドバーから「Incoming Webhooks」を選択")
    print("7. 「Activate Incoming Webhooks」をONにする")
    print("8. 「Add New Webhook to Workspace」をクリック")
    print("9. 投稿したいチャンネルを選択して「Allow」")
    print("10. 生成されたWebhook URLをコピー")
    print()
    print("💡 環境変数設定コマンド:")
    print("   export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'")
    print()
    print("💡 永続的な設定（bash）:")
    print("   echo 'export SLACK_WEBHOOK_URL=\"https://hooks.slack.com/services/YOUR/WEBHOOK/URL\"' >> ~/.bashrc")
    print("   source ~/.bashrc")
    print()
    print("💡 永続的な設定（zsh）:")
    print("   echo 'export SLACK_WEBHOOK_URL=\"https://hooks.slack.com/services/YOUR/WEBHOOK/URL\"' >> ~/.zshrc")
    print("   source ~/.zshrc")


def test_webhook_connection(webhook_url: str) -> bool:
    """Webhook URLの接続テスト"""
    print(f"\n🧪 Webhook接続テスト中...")
    
    test_payload = {
        "text": "🧪 LangGraph Workflow 設定テスト",
        "username": "Setup Helper Bot",
        "icon_emoji": ":gear:",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Slack統合セットアップ完了！* :tada:\n\nLangGraph Workflowからの通知が正常に動作しています。"
                }
            }
        ]
    }
    
    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(test_payload),
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        if response.status_code == 200:
            print("✅ テスト通知の送信が成功しました！")
            print("   Slackチャンネルで通知を確認してください")
            return True
        else:
            print(f"❌ テスト通知が失敗しました: {response.status_code}")
            print(f"   エラーメッセージ: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 接続エラー: {e}")
        return False


def interactive_setup():
    """対話的なセットアップ"""
    print("\n🔧 対話的セットアップを開始します")
    
    while True:
        webhook_url = input("\n📝 Slack Webhook URLを入力してください: ")
        
        if not webhook_url.strip():
            print("❌ URLが入力されていません")
            continue
            
        if not webhook_url.startswith("https://hooks.slack.com/"):
            print("❌ 無効なWebhook URL形式です")
            print("   正しい形式: https://hooks.slack.com/services/...")
            continue
            
        # テスト送信
        if test_webhook_connection(webhook_url):
            print(f"\n✅ Webhook URLが正常に動作しています！")
            
            # 環境変数設定の提案
            print("\n💡 環境変数を設定するには、以下のコマンドを実行してください:")
            print(f"   export SLACK_WEBHOOK_URL='{webhook_url}'")
            
            # .envファイルに保存するかどうか確認
            save_to_env = input("\n💾 .envファイルに保存しますか？ (y/N): ").lower().strip()
            if save_to_env in ['y', 'yes']:
                try:
                    with open('.env', 'a') as f:
                        f.write(f"\nSLACK_WEBHOOK_URL={webhook_url}\n")
                    print("✅ .envファイルに保存しました")
                except Exception as e:
                    print(f"❌ .envファイルの保存に失敗: {e}")
            
            return True
        else:
            retry = input("🔄 別のURLを試しますか？ (y/N): ").lower().strip()
            if retry not in ['y', 'yes']:
                break
    
    return False


def main():
    """メイン実行関数"""
    print_banner()
    
    # 現在の設定をチェック
    current_url = check_current_setup()
    
    if current_url:
        # 既に設定されている場合はテスト
        test_result = test_webhook_connection(current_url)
        if test_result:
            print("\n🎉 Slack統合は正常に動作しています！")
            return 0
        else:
            print("\n⚠️ 既存の設定で問題があります")
            
    # 設定手順を表示
    provide_setup_instructions()
    
    # 対話的セットアップを提案
    setup_now = input("\n🔧 今すぐセットアップしますか？ (y/N): ").lower().strip()
    
    if setup_now in ['y', 'yes']:
        if interactive_setup():
            print("\n🎉 セットアップが完了しました！")
            print("   LangGraph Workflowを実行してSlack通知をテストしてください")
            return 0
        else:
            print("\n⚠️ セットアップが完了しませんでした")
            return 1
    else:
        print("\n📋 後でセットアップする場合は、上記の手順に従ってください")
        return 0


if __name__ == "__main__":
    exit(main())
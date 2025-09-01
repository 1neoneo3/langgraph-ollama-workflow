"""Notification service for Slack integration."""

import json
import os
import time
import traceback
from typing import Dict, Tuple

from ..config.settings import Config
from ..core.state import WorkflowState


def validate_slack_webhook_url(webhook_url: str) -> Tuple[bool, str]:
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
    try:
        import requests
    except ImportError:
        print("❌ requests library not available for Slack notification")
        print("💡 Install with: pip install requests")
        return False

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

    except Exception as e:
        print(f"❌ Unexpected error sending Slack notification: {e}")
        traceback.print_exc()
        return {**state, "slack_notification_sent": False}
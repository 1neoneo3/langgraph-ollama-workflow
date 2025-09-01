"""LLM service for Ollama integration."""

import requests
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama

from ..config.settings import Config


def create_ollama_llm():
    """Create and return configured Ollama LLM instance."""
    return ChatOllama(
        model=Config.OLLAMA_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        temperature=Config.OLLAMA_TEMPERATURE,
    )


def handle_ollama_fallback(messages: List[BaseMessage], iteration: int) -> Dict[str, any]:
    """Handle Ollama fallback when service is unavailable."""
    if messages and isinstance(messages[-1], HumanMessage):
        content = messages[-1].content
        fallback_response = (
            f"Processing iteration {iteration}: {content} (Ollama unavailable)"
        )
        messages.append(AIMessage(content=fallback_response))

        return {
            "messages": messages,
            "processed_output": fallback_response,
        }
    return {"messages": messages}


def check_ollama_connection() -> bool:
    """Check if Ollama is running and the configured model is available."""
    try:
        print("🔍 Checking Ollama connection...")

        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)

        if response.status_code != 200:
            print(f"❌ Ollama API returned error: {response.status_code}")
            return False

        models = response.json()
        model_names = [model["name"] for model in models.get("models", [])]

        print(f"✅ Ollama is running with {len(model_names)} models")

        if Config.OLLAMA_MODEL in model_names:
            print(f"✅ {Config.OLLAMA_MODEL} model is available")
            return True
        else:
            print(f"❌ {Config.OLLAMA_MODEL} model not found")
            print("Available models:", model_names)
            print(
                f"\n💡 To install {Config.OLLAMA_MODEL}, run: ollama pull {Config.OLLAMA_MODEL}"
            )
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\n💡 Make sure Ollama is running: ollama serve")
        return False
    except ImportError:
        print("❌ requests library not available for Ollama check")
        return False
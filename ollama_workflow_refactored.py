#!/usr/bin/env python3
"""LangGraph workflow implementation with Ollama gpt-oss:20b model - Refactored version."""

from dotenv import load_dotenv
from src.main import main

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    exit(main())
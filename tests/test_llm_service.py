"""Tests for LLM service."""

import pytest
from unittest.mock import patch, Mock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.services.llm import (
    create_ollama_llm,
    handle_ollama_fallback,
    check_ollama_connection
)


class TestLLMService:
    """Test cases for LLM service functions."""

    def test_create_ollama_llm(self):
        """Test Ollama LLM creation."""
        with patch('src.services.llm.ChatOllama') as mock_chat_ollama:
            mock_instance = MagicMock()
            mock_chat_ollama.return_value = mock_instance
            
            result = create_ollama_llm()
            
            # Check that ChatOllama was called
            mock_chat_ollama.assert_called_once()
            
            # Check that the call included expected parameters
            call_args = mock_chat_ollama.call_args[1]  # keyword arguments
            assert 'model' in call_args
            assert 'base_url' in call_args
            assert 'temperature' in call_args
            
            assert result == mock_instance

    def test_handle_ollama_fallback_with_human_message(self):
        """Test Ollama fallback with HumanMessage."""
        messages = [HumanMessage(content="テスト質問")]
        iteration = 1
        
        result = handle_ollama_fallback(messages, iteration)
        
        # Check return structure
        assert "messages" in result
        assert "processed_output" in result
        
        # Check that AIMessage was added
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][1], AIMessage)
        
        # Check fallback content
        ai_message_content = result["messages"][1].content
        assert "テスト質問" in ai_message_content
        assert str(iteration) in ai_message_content
        assert "Ollama unavailable" in ai_message_content
        
        # Check processed_output matches
        assert result["processed_output"] == ai_message_content

    def test_handle_ollama_fallback_without_human_message(self):
        """Test Ollama fallback without HumanMessage."""
        messages = [AIMessage(content="AI response")]
        iteration = 1
        
        result = handle_ollama_fallback(messages, iteration)
        
        # Should return messages as is
        assert "messages" in result
        assert result["messages"] == messages
        assert "processed_output" not in result

    def test_handle_ollama_fallback_empty_messages(self):
        """Test Ollama fallback with empty messages."""
        messages = []
        iteration = 1
        
        result = handle_ollama_fallback(messages, iteration)
        
        assert "messages" in result
        assert result["messages"] == messages

    @patch('src.services.llm.requests')
    def test_check_ollama_connection_success(self, mock_requests):
        """Test successful Ollama connection check."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1"},
                {"name": "codellama"}
            ]
        }
        mock_requests.get.return_value = mock_response
        
        # Mock Config to include expected model
        with patch('src.services.llm.Config') as mock_config:
            mock_config.OLLAMA_MODEL = "llama3.1"
            mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
            
            result = check_ollama_connection()
            
            assert result is True
            mock_requests.get.assert_called_once_with(
                "http://localhost:11434/api/tags", timeout=5
            )

    @patch('src.services.llm.requests')
    def test_check_ollama_connection_model_not_found(self, mock_requests):
        """Test Ollama connection when model is not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "other-model"}
            ]
        }
        mock_requests.get.return_value = mock_response
        
        with patch('src.services.llm.Config') as mock_config:
            mock_config.OLLAMA_MODEL = "llama3.1"
            mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
            
            result = check_ollama_connection()
            
            assert result is False

    @patch('src.services.llm.requests')
    def test_check_ollama_connection_api_error(self, mock_requests):
        """Test Ollama connection with API error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_requests.get.return_value = mock_response
        
        with patch('src.services.llm.Config') as mock_config:
            mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
            
            result = check_ollama_connection()
            
            assert result is False

    @patch('src.services.llm.requests')
    def test_check_ollama_connection_request_exception(self, mock_requests):
        """Test Ollama connection with request exception."""
        # Create a proper RequestException instance
        from requests.exceptions import RequestException
        mock_requests.exceptions.RequestException = RequestException
        mock_requests.get.side_effect = RequestException("Connection failed")
        
        with patch('src.services.llm.Config') as mock_config:
            mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
            
            result = check_ollama_connection()
            
            assert result is False

    @patch('src.services.llm.requests', side_effect=ImportError)
    def test_check_ollama_connection_import_error(self, mock_requests):
        """Test Ollama connection with ImportError."""
        result = check_ollama_connection()
        assert result is False

    def test_handle_ollama_fallback_multiple_messages(self):
        """Test Ollama fallback with multiple messages."""
        messages = [
            AIMessage(content="Previous AI response"),
            HumanMessage(content="New question"),
        ]
        iteration = 2
        
        result = handle_ollama_fallback(messages, iteration)
        
        # Should have added one more AIMessage
        assert len(result["messages"]) == 3
        assert isinstance(result["messages"][-1], AIMessage)
        
        # Check the last message content
        last_message = result["messages"][-1]
        assert "New question" in last_message.content
        assert str(iteration) in last_message.content
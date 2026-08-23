import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from agent_cli.auth import get_stored_credentials, save_credentials
from agent_cli.main import setup_provider_and_auth, main

# ==========================================
# 1. AUTH & CREDENTIAL STORAGE TESTS
# ==========================================

def test_save_and_get_credentials(tmp_path, monkeypatch):
    test_credentials_file = tmp_path / "credentials.json"
    monkeypatch.setattr("agent_cli.auth.CREDENTIALS_FILE", test_credentials_file)
    monkeypatch.setattr("agent_cli.auth.CONFIG_DIR", tmp_path)

    assert get_stored_credentials("claude") is None

    save_credentials("claude", "sk-ant-test-key-123")
    assert get_stored_credentials("claude") == "sk-ant-test-key-123"

    save_credentials("gemini", "gemini-test-key-456")
    assert get_stored_credentials("claude") == "sk-ant-test-key-123"
    assert get_stored_credentials("gemini") == "gemini-test-key-456"

# ==========================================
# 2. INTERACTIVE WIZARD TESTS
# ==========================================

@patch("agent_cli.main.get_stored_credentials")
@patch("rich.prompt.Prompt.ask")
def test_setup_provider_and_auth_ollama(mock_ask, mock_get_credentials):
    # Mock user choosing provider '1' (Ollama) and model '1'
    mock_ask.side_effect = ["1", "1"]

    provider, model, api_key = setup_provider_and_auth()

    assert provider == "ollama"
    assert model == "qwen2.5-coder:7b-instruct"
    assert api_key is None
    mock_get_credentials.assert_not_called()

@patch("agent_cli.main.save_credentials")
@patch("agent_cli.main.get_stored_credentials")
@patch("rich.prompt.Prompt.ask")
def test_setup_provider_and_auth_claude_new_key(mock_ask, mock_get_cred, mock_save_cred):
    # Mock no stored credentials found 
    mock_get_cred.return_value = None
    # User inputs: provider '3' (Claude), model '1', API key "sk-test"
    mock_ask.side_effect = ["3", "1", "sk-test"]

    provider, model, api_key = setup_provider_and_auth()

    assert provider == "claude"
    assert model == "claude-3-5-sonnet-20241022"
    assert api_key == "sk-test"
    mock_save_cred.assert_called_once_with("claude", "sk-test")

# ==========================================
# 3. REPL MAIN LOOP TESTS
# ==========================================

@patch("agent_cli.main.setup_provider_and_auth")
@patch("agent_cli.main.create_llm_client")
@patch("agent_cli.main.VectorStoreManager")
@patch("rich.prompt.Prompt.ask")
def test_main_repl_loop_execution(mock_prompt, mock_vector, mock_create_llm, mock_setup):
    # 1. Setup mock provider configuration
    mock_setup.return_value = ("ollama", "qwen2.5-coder:7b-instruct", None)

    # 2. Setup mock LLM Cllient and Vector Store
    mock_llm_instance = MagicMock()
    mock_llm_instance.chat = AsyncMock(return_value=("Done!", {"provider": "ollama", "input_tokens": 10}))
    mock_create_llm.return_value = mock_llm_instance

    mock_vector_instance = MagicMock()
    mock_vector_instance.search_code_base.return_value = [{"file_path": "main.py", "content": "print('hello')"}]
    mock_vector.return_value = mock_vector_instance

    # 3. User prompts: first query then 'exit'
    mock_prompt.side_effect = ["How does this work?", "exit"]

    # Run main loop
    main()

    # Assertions
    mock_vector_instance.search_codebase.assert_called_once_with("How does this work?", top_k=2)
    mock_llm_instance.chat.assert_called_once()
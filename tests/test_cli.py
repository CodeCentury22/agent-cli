import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from agent_cli.auth import get_stored_credentials, save_credentials
from agent_cli.agent_config import setup_provider_and_auth
from agent_cli.main import main
from agent_cli.agent_workspace import initialize_workspace_vector_memory
from agent_cli.agent_orchestrator import run_agent_turn, parse_tool_call

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

@patch("agent_cli.agent_config.get_stored_credentials")
@patch("rich.prompt.Prompt.ask")
def test_setup_provider_and_auth_ollama(mock_ask, mock_get_credentials):
    mock_ask.side_effect = ["1", "1"]

    provider, model, api_key = setup_provider_and_auth()

    assert provider == "ollama"
    assert model == "qwen2.5-coder:7b-instruct"
    assert api_key is None
    mock_get_credentials.assert_not_called()


@patch("agent_cli.agent_config.save_credentials")
@patch("agent_cli.agent_config.get_stored_credentials")
@patch("rich.prompt.Prompt.ask")
def test_setup_provider_and_auth_claude_new_key(mock_ask, mock_get_cred, mock_save_cred):
    mock_get_cred.return_value = None
    mock_ask.side_effect = ["3", "1", "sk-test"]

    provider, model, api_key = setup_provider_and_auth()

    assert provider == "claude"
    assert model == "claude-3-5-sonnet-20241022"
    assert api_key == "sk-test"
    mock_save_cred.assert_called_once_with("claude", "sk-test")


# ==========================================
# 3. WORKSPACE VECTOR MEMORY TESTS
# ==========================================

@pytest.mark.asyncio
@patch("agent_cli.agent_workspace.sync_workspace_vector_memory")
@patch("agent_cli.agent_workspace.get_git_status_changes", new_callable=AsyncMock)
async def test_initialize_workspace_vector_memory_sync(mock_git_status, mock_sync_memory, tmp_path):
    mock_vector_store = MagicMock()
    mock_git_status.return_value = (True, {"src/app.ts"}, {"src/old.ts"})

    await initialize_workspace_vector_memory(mock_vector_store, str(tmp_path))

    mock_git_status.assert_called_once_with(str(tmp_path.resolve()))
    mock_sync_memory.assert_called_once_with(
        vector_store=mock_vector_store,
        workspace_dir=str(tmp_path.resolve()),
        is_git_repo=True,
        files_to_update={"src/app.ts"},
        files_to_delete={"src/old.ts"}
    )


# ==========================================
# 4. REPL MAIN LOOP TESTS
# ==========================================

@patch("agent_cli.main.initialize_workspace_vector_memory", new_callable=AsyncMock)
@patch("agent_cli.main.ensure_preset_skills_exist")
@patch("agent_cli.main.ensure_agent_gitignore_entries")
@patch("agent_cli.main.setup_provider_and_auth")
@patch("agent_cli.main.create_llm_client")
@patch("agent_cli.main.VectorStoreManager")
@patch("agent_cli.main.run_agent_turn", new_callable=AsyncMock)
@patch("agent_cli.main.PromptSession.prompt_async", new_callable=AsyncMock)
def test_main_repl_loop_execution(
    mock_prompt_async,
    mock_run_turn,
    mock_vector_class,
    mock_create_llm,
    mock_setup,
    mock_gitignore,
    mock_preset_skills,
    mock_init_vector_memory
):
    mock_setup.return_value = ("ollama", "qwen2.5-coder:7b-instruct", None)
    
    mock_llm_instance = AsyncMock()
    mock_llm_instance.ensure_model_available.return_value = True
    mock_create_llm.return_value = mock_llm_instance
    
    mock_vector_instance = MagicMock()
    mock_vector_class.return_value = mock_vector_instance

    mock_prompt_async.side_effect = ["How does this work?", "exit"]

    main()

    mock_gitignore.assert_called_once()
    mock_preset_skills.assert_called_once()
    mock_setup.assert_called_once()
    mock_init_vector_memory.assert_called_once_with(mock_vector_instance, ".")
    mock_run_turn.assert_called_once_with("How does this work?", mock_llm_instance, mock_vector_instance)


# ==========================================
# 5. ORCHESTRATOR & PARSER TESTS
# ==========================================

def test_parse_tool_call_valid_json_string():
    response_str = '{"name": "run_shell_command", "arguments": {"command": "ls"}}'
    tool_name, args = parse_tool_call(response_str)
    assert tool_name == "run_shell_command"
    assert args == {"command": "ls"}


def test_parse_tool_call_sdk_object():
    mock_obj = MagicMock()
    mock_obj.tool_calls = [{"name": "read_file", "arguments": {"file_path": "src/main.ts"}}]
    tool_name, args = parse_tool_call(mock_obj)
    assert tool_name == "read_file"
    assert args == {"file_path": "src/main.ts"}


def test_parse_tool_call_markdown_code_block():
    response_str = (
        "Here is the tool call to update the file:\n\n"
        "```json\n"
        "{\n"
        '  "name": "write_file",\n'
        '  "arguments": {\n'
        '    "file_path": "src/app/login.css",\n'
        '    "code_body": ".card { color: blue; }"\n'
        "  }\n"
        "}\n"
        "```"
    )
    tool_name, args = parse_tool_call(response_str)
    assert tool_name == "write_file"
    assert args == {
        "file_path": "src/app/login.css",
        "code_body": ".card { color: blue; }"
    }


def test_parse_tool_call_embedded_in_text():
    response_str = (
        "Writing styles now. "
        '{"tool_name": "write_file", "arguments": {"path": "styles.css", "content": "body{}"}} '
        "Done."
    )
    tool_name, args = parse_tool_call(response_str)
    assert tool_name == "write_file"
    assert args == {"path": "styles.css", "content": "body{}"}


@pytest.mark.asyncio
@patch("agent_cli.agent_orchestrator.validate_tool_args")
@patch("agent_cli.agent_orchestrator.handle_tool_call", new_callable=AsyncMock)
async def test_run_agent_turn_sliding_window_circuit_breaker(mock_handle_tool, mock_validate):
    """Verify circuit breaker catches alternating duplicate tool calls across turns."""
    mock_llm_client = AsyncMock()
    mock_vector_store = MagicMock()
    mock_vector_store.search_codebase.return_value = []

    # Mock validate_tool_args to return valid status and arguments directly
    mock_validate.side_effect = lambda name, args: (True, args, "")

    # Sequence: cmd_a -> cmd_b -> cmd_a -> cmd_a (triggers circuit breaker on Turn 4)
    cmd_a = '{"name": "run_shell_command", "arguments": {"command": "which ng"}}'
    cmd_b = '{"name": "run_shell_command", "arguments": {"command": "ng version"}}'
    
    mock_llm_client.chat.side_effect = [
        (cmd_a, {"input_tokens": 10, "output_tokens": 5}),  # Turn 1: cmd_a executed
        (cmd_b, {"input_tokens": 10, "output_tokens": 5}),  # Turn 2: cmd_b executed
        (cmd_a, {"input_tokens": 10, "output_tokens": 5}),  # Turn 3: cmd_a executed
        (cmd_a, {"input_tokens": 10, "output_tokens": 5}),  # Turn 4: chat called, circuit breaker halts turn
    ]
    
    mock_handle_tool.return_value = "Command output ok"

    await run_agent_turn("Check ng environment", mock_llm_client, mock_vector_store)

    # handle_tool_call should be invoked exactly 3 times before Turn 4 halts execution
    assert mock_handle_tool.call_count == 3
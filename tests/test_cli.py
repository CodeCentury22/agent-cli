import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from agent_cli.auth import get_stored_credentials, save_credentials
from agent_cli.agent_config import setup_provider_and_auth
from agent_cli.main import main
from agent_cli.mcp_manager import (
    ensure_and_load_mcp_servers,
    get_mcp_tool_schemas_and_dispatchers,
)
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
# 2. MCP MANAGER TESTS
# ==========================================

def test_ensure_and_load_mcp_servers_creates_template(tmp_path, monkeypatch):
    """Verify .agent/mcp.json template creation and default guidance behavior."""
    monkeypatch.chdir(tmp_path)

    with patch("agent_cli.mcp_manager.display_mcp_guidance") as mock_guidance:
        active_servers = ensure_and_load_mcp_servers()

        mcp_file = tmp_path / ".agent" / "mcp.json"
        assert mcp_file.exists()
        assert active_servers == {}
        mock_guidance.assert_called_once()


def test_ensure_and_load_mcp_servers_parses_configured_servers(tmp_path, monkeypatch):
    """Verify configured non-comment MCP servers are loaded properly."""
    monkeypatch.chdir(tmp_path)
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "mcpServers": {
            "// comment": {"command": "disabled"},
            "git": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-git"]}
        }
    }
    (agent_dir / "mcp.json").write_text(json.dumps(config_data), encoding="utf-8")

    with patch("agent_cli.mcp_manager.display_mcp_guidance") as mock_guidance:
        active_servers = ensure_and_load_mcp_servers()

        assert "git" in active_servers
        assert "// comment" not in active_servers
        mock_guidance.assert_not_called()


@pytest.mark.asyncio
async def test_get_mcp_tool_schemas_and_dispatchers_empty(tmp_path, monkeypatch):
    """Verify get_mcp_tool_schemas_and_dispatchers returns empty tuple when no servers active."""
    monkeypatch.chdir(tmp_path)

    with patch("agent_cli.mcp_manager.display_mcp_guidance"):
        schemas, dispatchers = await get_mcp_tool_schemas_and_dispatchers()
        assert schemas == []
        assert dispatchers == {}


# ==========================================
# 3. INTERACTIVE WIZARD TESTS
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


# ==========================================
# 4. REPL MAIN LOOP TESTS
# ==========================================

@patch("agent_cli.main.initialize_workspace_vector_memory", new_callable=AsyncMock)
@patch("agent_cli.main.ensure_and_load_mcp_servers")
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
    mock_mcp_servers,
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
    mock_mcp_servers.assert_called_once()
    mock_setup.assert_called_once()
    mock_init_vector_memory.assert_called_once_with(mock_vector_instance, ".")
    mock_run_turn.assert_called_once_with("How does this work?", mock_llm_instance, mock_vector_instance)


# ==========================================
# 5. ORCHESTRATOR & MCP DYNAMIC MERGE TESTS
# ==========================================

@pytest.mark.asyncio
@patch("agent_cli.agent_orchestrator.get_mcp_tool_schemas_and_dispatchers", new_callable=AsyncMock)
@patch("agent_cli.agent_orchestrator.validate_tool_args")
@patch("agent_cli.agent_orchestrator.handle_tool_call", new_callable=AsyncMock)
async def test_run_agent_turn_circuit_breaker(mock_handle_tool, mock_validate, mock_get_mcp):
    """Verify run_agent_turn syncs git changes, resolves dynamic MCP schemas, and executes tool loops."""
    # Return empty MCP schemas and dispatchers for default execution
    mock_get_mcp.return_value = ([], {})
    
    mock_llm_client = AsyncMock()
    mock_vector_store = MagicMock()
    mock_vector_store.sync_git_changes.return_value = 0
    mock_vector_store.search_codebase.return_value = []

    mock_validate.side_effect = lambda name, args: (True, args, "")

    cmd_a = '{"name": "run_shell_command", "arguments": {"command": "which ng"}}'
    
    mock_llm_client.chat.side_effect = [
        (cmd_a, {"input_tokens": 10, "output_tokens": 5}),
        (cmd_a, {"input_tokens": 10, "output_tokens": 5}),
        (cmd_a, {"input_tokens": 10, "output_tokens": 5}),
    ]
    
    mock_handle_tool.return_value = "Command output ok"

    await run_agent_turn("Check environment", mock_llm_client, mock_vector_store)

    # 1. Verify workspace git changes were synced into Chroma prior to vector search
    mock_vector_store.sync_git_changes.assert_called_once_with(root_dir=".")

    # 2. Verify MCP schemas were requested during turn initialization
    mock_get_mcp.assert_called_once()
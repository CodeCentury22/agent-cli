import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agent_cli.skill_downloader import (
    ensure_preset_skills_exist,
    find_and_parse_skill_lock,
    has_existing_workspace_skills,
)
from agent_cli.agent_workspace import load_project_skills, ensure_agent_gitignore_entries
from agent_cli.agent_orchestrator import (
    format_memory_results,
    parse_tool_call,
    run_agent_turn,
    sanitize_readme_for_llm,
    read_workspace_readme,
)
from agent_cli.tool_handler import handle_tool_call


# =====================================================================
# Tests for tool_handler.py & Guardrails
# =====================================================================

@pytest.mark.asyncio
async def test_handle_tool_call_successful_execution():
    """Verify handle_tool_call validates arguments and dispatches correctly."""
    mock_dispatcher = MagicMock(return_value={"status": "SUCCESS", "output": "file contents"})
    dispatchers = {"read_file": mock_dispatcher}

    with patch("agent_cli.tool_handler.validate_tool_args", return_value=(True, {"file_path": "test.py"}, None)):
        result = await handle_tool_call("read_file", {"file_path": "test.py"}, dispatchers)

        assert result == {"status": "SUCCESS", "output": "file contents"}
        mock_dispatcher.assert_called_once_with(file_path="test.py")


@pytest.mark.asyncio
async def test_handle_tool_call_guardrail_rejection():
    """Verify handle_tool_call returns error when guardrails reject arguments."""
    dispatchers = {"run_shell_command": MagicMock()}

    with patch("agent_cli.tool_handler.validate_tool_args", return_value=(False, {}, "Forbidden command")):
        result = await handle_tool_call("run_shell_command", {"command": "rm -rf /"}, dispatchers)

        assert result["status"] == "ERROR"
        assert result["error"] == "Forbidden command"
        dispatchers["run_shell_command"].assert_not_called()


@pytest.mark.asyncio
async def test_handle_tool_call_unregistered_tool():
    """Verify handle_tool_call handles tools not present in dispatchers dict."""
    dispatchers = {}

    with patch("agent_cli.tool_handler.validate_tool_args", return_value=(True, {}, None)):
        result = await handle_tool_call("unknown_tool", {}, dispatchers)

        assert result["status"] == "ERROR"
        assert "not registered" in result["error"]


@pytest.mark.asyncio
async def test_handle_tool_call_dispatcher_exception_is_wrapped():
    """Verify dispatcher exceptions are converted to ERROR payloads, not raised."""
    async def _boom(**kwargs):
        raise RuntimeError("kaboom")

    dispatchers = {"run_shell_command": _boom}

    with patch("agent_cli.tool_handler.validate_tool_args", return_value=(True, {"command": "echo hi"}, None)):
        result = await handle_tool_call("run_shell_command", {"command": "echo hi"}, dispatchers)

        assert result["status"] == "ERROR"
        assert "kaboom" in result["error"]


# =====================================================================
# Tests for skill_downloader.py (Lockfiles & Discovery)
# =====================================================================

def test_find_and_parse_skill_lock(tmp_path):
    """Verify parsing of skill-lock.json structure."""
    lock_data = {
        "skills": {
            "angular-developer": {
                "source": "angular/skills",
                "sourceType": "github",
                "skillPath": "angular-developer/SKILL.md",
            }
        }
    }
    lock_file = tmp_path / "skill-lock.json"
    lock_file.write_text(json.dumps(lock_data), encoding="utf-8")

    parsed = find_and_parse_skill_lock(str(tmp_path))

    assert "angular-developer" in parsed
    assert parsed["angular-developer"]["raw_url"] == "https://raw.githubusercontent.com/angular/skills/main/angular-developer/SKILL.md"
    assert parsed["angular-developer"]["lock_file"] == "skill-lock.json"


def test_has_existing_workspace_skills(tmp_path):
    """Verify detection of existing rules/skills in workspace."""
    cursor_rules = tmp_path / ".cursor" / "rules"
    cursor_rules.mkdir(parents=True, exist_ok=True)
    (cursor_rules / "python.mdc").write_text("Rule content")

    assert has_existing_workspace_skills(str(tmp_path)) is True


def test_ensure_preset_skills_exist_skips_when_skills_exist(tmp_path, monkeypatch):
    """Verify preset downloading is skipped if existing skills are discovered."""
    monkeypatch.chdir(tmp_path)
    skills_dir = tmp_path / ".agent" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "custom.md").write_text("# Custom Skill")

    with patch("httpx.Client") as mock_httpx, patch("subprocess.run") as mock_git:
        ensure_preset_skills_exist()

        mock_httpx.assert_not_called()
        mock_git.assert_not_called()
        assert (skills_dir / ".preset_installed").exists()


def test_ensure_preset_skills_exist_syncs_lockfile(tmp_path, monkeypatch):
    """Verify downloading from lockfile takes priority over platform detection."""
    monkeypatch.chdir(tmp_path)
    lock_data = {
        "skills": {
            "python-fastapi": {
                "source": "PatrickJS/awesome-cursorrules",
                "sourceType": "github",
                "skillPath": "rules/python.mdc",
            }
        }
    }
    (tmp_path / "skills-lock.json").write_text(json.dumps(lock_data), encoding="utf-8")

    with patch("httpx.Client.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.text = "# FastAPI Rules"
        mock_get.return_value = mock_res

        ensure_preset_skills_exist()

    target_skill = tmp_path / ".agent" / "skills" / "python-fastapi.md"
    assert target_skill.exists()
    assert target_skill.read_text() == "# FastAPI Rules"


# =====================================================================
# Tests for agent_workspace.py
# =====================================================================

def test_load_project_skills_merges_markdown_and_ignores_dotfiles(tmp_path):
    """Verify load_project_skills discovers .md files and ignores sentinel files."""
    skills_dir = tmp_path / ".agent" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    (skills_dir / "python.md").write_text("Python Best Practices")
    (skills_dir / ".preset_installed").write_text("installed")

    with patch("os.getcwd", return_value=str(tmp_path)), \
         patch("agent_cli.agent_workspace.ensure_preset_skills_exist"):

        result = load_project_skills()

        assert "PROJECT SKILLS & DOMAIN GUIDELINES" in result
        assert "python.md" in result
        assert "Python Best Practices" in result
        assert ".preset_installed" not in result


def test_ensure_agent_gitignore_entries_appends_missing_patterns(tmp_path):
    """Verify gitignore entries are appended correctly via workspace tools."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("# Existing gitignore\nnode_modules/\n")

    async def _fake_shell(command, timeout=10.0, bypass_hitl=False):
        _fake_shell.calls.append(command)
        return {"returncode": 0, "status": "SUCCESS", "stdout": "", "stderr": ""}

    _fake_shell.calls = []

    with patch("os.getcwd", return_value=str(tmp_path)), \
         patch("agent_cli.agent_workspace.execute_async_subprocess", side_effect=_fake_shell):

        import asyncio
        asyncio.run(ensure_agent_gitignore_entries())

        content = gitignore.read_text()
        assert "# Agent CLI auto-generated artifacts" in content
        assert ".agent/skills/" in content
        assert "*.jsonl" in content
        assert any("git add .gitignore" in c for c in _fake_shell.calls)


# =====================================================================
# Tests for README Sanitization & Orchestrator Parsing
# =====================================================================

def test_sanitize_readme_for_llm():
    """Verify images, badge links, and HTML comments are stripped from README."""
    raw_readme = (
        "# Project Title\n"
        "![Banner](https://example.com/banner.png)\n"
        "<!-- HTML Comment -->\n"
        "<img src='badge.svg' />\n"
        "## Setup\nUse `pnpm install`."
    )
    sanitized = sanitize_readme_for_llm(raw_readme)

    assert "![Banner]" not in sanitized
    assert "<!-- HTML Comment -->" not in sanitized
    assert "<img" not in sanitized
    assert "## Setup\nUse `pnpm install`." in sanitized


def test_read_workspace_readme(tmp_path):
    """Verify read_workspace_readme locates and sanitizes README.md."""
    readme = tmp_path / "README.md"
    readme.write_text("# Sample Project\nRun `npm test`.", encoding="utf-8")

    result = read_workspace_readme(str(tmp_path))

    assert "Sample Project" in result
    assert "Run `npm test`." in result


def test_parse_tool_call_object_format():
    """Verify parsing tool calls from SDK object responses."""
    mock_obj = MagicMock()
    mock_obj.tool_calls = [{"name": "write_file", "arguments": {"file_path": "main.py"}}]

    name, args = parse_tool_call(mock_obj)
    assert name == "write_file"
    assert args == {"file_path": "main.py"}


def test_parse_tool_call_json_string_format():
    """Verify parsing tool calls from raw JSON string responses."""
    raw_json = '{"name": "read_file", "arguments": {"file_path": "package.json"}}'

    name, args = parse_tool_call(raw_json)
    assert name == "read_file"
    assert args == {"file_path": "package.json"}


def test_parse_tool_call_non_tool_response():
    """Verify returning None for plain text responses without tool requests."""
    text_response = "Here is the refactored code for your Angular component."

    name, args = parse_tool_call(text_response)
    assert name is None
    assert args == {}


def test_format_memory_results_uses_enriched_metadata():
    """Verify enriched vector-memory metadata reaches the orchestrator context."""
    matches = [
        {
            "file_path": "src/app.py",
            "content": "def handler(): pass",
            "relevance_score": 0.85,
            "distance": 0.15,
            "start_line": 10,
            "end_line": 20,
            "language": "Python",
            "heading": "handler",
        },
        {
            "file_path": "src/other.py",
            "content": "x = 1",
            "relevance_score": 0.5,
            "distance": 0.5,
            "start_line": None,
            "end_line": None,
            "language": "",
            "heading": "",
        },
    ]

    formatted = format_memory_results(matches)

    assert "src/app.py#L10-20" in formatted
    assert "lang=Python" in formatted
    assert "symbol=handler" in formatted
    assert "relevance=0.85" in formatted
    assert "def handler(): pass" in formatted
    assert "src/other.py" in formatted
    assert format_memory_results([]) == ""


@pytest.mark.asyncio
async def test_ensure_agent_gitignore_entries_skips_when_complete(tmp_path):
    """Verify no git writes happen when all ignore patterns already exist."""
    from agent_cli.agent_workspace import AGENT_IGNORES

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    existing = "# Agent CLI auto-generated artifacts\n" + "\n".join(AGENT_IGNORES) + "\n"
    (tmp_path / ".gitignore").write_text(existing)

    with patch("os.getcwd", return_value=str(tmp_path)), \
         patch("agent_cli.agent_workspace.execute_async_subprocess", new_callable=AsyncMock) as mock_shell:
        await ensure_agent_gitignore_entries()
        mock_shell.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_turn_hallucinated_tool_triggers_repair():
    """Verify unknown hallucinated tools are intercepted with a repair nudge."""
    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = [
        ('{"name": "teleport_file", "arguments": {}}', {"input_tokens": 1, "output_tokens": 1}),
        ('{"name": "done", "arguments": {}}', {"input_tokens": 1, "output_tokens": 1}),
    ]
    mock_store = MagicMock()
    mock_store.search_codebase.return_value = []

    with patch("agent_cli.agent_orchestrator.get_git_status_changes", new_callable=AsyncMock) as mock_git, \
         patch("agent_cli.agent_orchestrator.sync_workspace_vector_memory") as mock_sync, \
         patch("agent_cli.agent_orchestrator.get_mcp_tool_schemas_and_dispatchers", new_callable=AsyncMock) as mock_mcp, \
         patch("agent_cli.agent_orchestrator.load_project_skills", return_value=""), \
         patch("agent_cli.agent_orchestrator.read_workspace_readme", return_value=""), \
         patch("agent_cli.agent_orchestrator.load_workspace_config_context", return_value=""):
        mock_git.return_value = (False, set(), set())
        mock_mcp.return_value = ([], {})

        await run_agent_turn("do work", mock_llm, mock_store)

        assert mock_llm.chat.call_count == 2
        repair_prompt = mock_llm.chat.call_args[0][0][-1]["content"]
        assert "not registered" in repair_prompt
        mock_store.sync_git_changes.assert_not_called()
        mock_sync.assert_called_once()


@pytest.mark.asyncio
async def test_run_agent_turn_repeating_tool_triggers_circuit_breaker():
    """Verify repeating identical tool calls halt the turn instead of looping."""
    mock_llm = AsyncMock()
    tool_json = '{"name": "read_file", "arguments": {"file_path": "a.py"}}'
    mock_llm.chat.side_effect = [(tool_json, {})] * 5
    mock_store = MagicMock()
    mock_store.search_codebase.return_value = []

    with patch("agent_cli.agent_orchestrator.get_git_status_changes", new_callable=AsyncMock) as mock_git, \
         patch("agent_cli.agent_orchestrator.sync_workspace_vector_memory"), \
         patch("agent_cli.agent_orchestrator.get_mcp_tool_schemas_and_dispatchers", new_callable=AsyncMock) as mock_mcp, \
         patch("agent_cli.agent_orchestrator.load_project_skills", return_value=""), \
         patch("agent_cli.agent_orchestrator.read_workspace_readme", return_value=""), \
         patch("agent_cli.agent_orchestrator.load_workspace_config_context", return_value=""), \
         patch("agent_cli.agent_orchestrator.handle_tool_call", new_callable=AsyncMock) as mock_handle:
        from agent_workspace_tools import WORKSPACE_TOOL_DISPATCHER
        assert "read_file" in WORKSPACE_TOOL_DISPATCHER
        mock_git.return_value = (False, set(), set())
        mock_mcp.return_value = ([], {})
        mock_handle.return_value = {"status": "SUCCESS"}

        await run_agent_turn("read it", mock_llm, mock_store)

        assert mock_handle.call_count == 2
        assert mock_llm.chat.call_count == 3


@pytest.mark.asyncio
async def test_run_agent_turn_validation_error_retries_then_halts():
    """Verify schema validation errors retry with repair prompts, then halt."""
    mock_llm = AsyncMock()
    bad_json = '{"name": "write_file", "arguments": {"file_path": 123}}'
    mock_llm.chat.side_effect = [(bad_json, {})] * 5
    mock_store = MagicMock()
    mock_store.search_codebase.return_value = []

    with patch("agent_cli.agent_orchestrator.get_git_status_changes", new_callable=AsyncMock) as mock_git, \
         patch("agent_cli.agent_orchestrator.sync_workspace_vector_memory"), \
         patch("agent_cli.agent_orchestrator.get_mcp_tool_schemas_and_dispatchers", new_callable=AsyncMock) as mock_mcp, \
         patch("agent_cli.agent_orchestrator.load_project_skills", return_value=""), \
         patch("agent_cli.agent_orchestrator.read_workspace_readme", return_value=""), \
         patch("agent_cli.agent_orchestrator.load_workspace_config_context", return_value=""), \
         patch("agent_cli.agent_orchestrator.validate_tool_args", return_value=(False, {}, "bad args")), \
         patch("agent_cli.agent_orchestrator.handle_tool_call", new_callable=AsyncMock) as mock_handle:
        mock_git.return_value = (False, set(), set())
        mock_mcp.return_value = ([], {})

        await run_agent_turn("write it", mock_llm, mock_store)

        mock_handle.assert_not_called()
        assert mock_llm.chat.call_count == 3
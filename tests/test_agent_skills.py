import pytest
from unittest.mock import patch, MagicMock
from agent_cli.skill_downloader import ensure_preset_skills_exist, PRESET_SKILLS
from agent_cli.agent_workspace import load_project_skills, ensure_agent_gitignore_entries
from agent_cli.agent_orchestrator import parse_tool_call
from agent_cli.tool_handler import handle_tool_call


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


# =====================================================================
# Tests for skill_downloader.py
# =====================================================================

def test_ensure_preset_skills_exist_bypasses_if_sentinel_exists(tmp_path):
    """Verify that if .preset_installed exists, downloading is skipped completely."""
    skills_dir = tmp_path / ".agent" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    sentinel = skills_dir / ".preset_installed"
    sentinel.write_text("installed")

    with patch("os.getcwd", return_value=str(tmp_path)), \
         patch("subprocess.run") as mock_git, \
         patch("httpx.Client") as mock_httpx:

        ensure_preset_skills_exist()

        mock_git.assert_not_called()
        mock_httpx.assert_not_called()


def test_ensure_preset_skills_exist_downloads_and_writes_sentinel(tmp_path, monkeypatch):
    # Temporarily switch the working directory to the pytest tmp_path
    monkeypatch.chdir(tmp_path)

    # Mock stack detection to return at least one platform (e.g. ['python'])
    with patch("agent_cli.skill_downloader.detect_project_platforms", return_value=["python"]), \
         patch("httpx.Client.get") as mock_get:
        
        # Mock successful HTTP 200 response for preset skill fetch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Python Skill Guidelines"
        mock_get.return_value = mock_response

        # Execute target function
        ensure_preset_skills_exist()

    # Assertions on local tmp_path directory structure
    skills_dir = tmp_path / ".agent" / "skills"
    sentinel = skills_dir / ".preset_installed"
    python_skill = skills_dir / "python.md"

    assert skills_dir.exists()
    assert sentinel.exists()
    assert python_skill.exists()
    assert python_skill.read_text() == "# Python Skill Guidelines"


# =====================================================================
# Tests for agent_workspace.py
# =====================================================================

def test_load_project_skills_merges_markdown_and_ignores_dotfiles(tmp_path):
    """Verify load_project_skills discovers .md files and ignores sentinel files."""
    skills_dir = tmp_path / ".agent" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy skill file and sentinel
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
    """Verify gitignore entries are appended correctly."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("# Existing gitignore\nnode_modules/\n")

    with patch("os.getcwd", return_value=str(tmp_path)), \
         patch("subprocess.run") as mock_git:

        ensure_agent_gitignore_entries()

        content = gitignore.read_text()
        assert "# Agent CLI auto-generated artifacts" in content
        assert ".agent/skills/" in content
        assert "*.jsonl" in content
        mock_git.assert_called()


# =====================================================================
# Tests for agent_orchestrator.py
# =====================================================================

def test_parse_tool_call_object_format():
    """Verify parsing tool calls from SDK object responses."""
    mock_obj = MagicMock()
    mock_obj.tool_calls = [{"name": "write_file", "arguments": {"file_path": "main.py"}}]

    name, args = parse_tool_call(mock_obj)
    assert name == "write_file"
    assert args == {"file_path": "main.py"}


def test_parse_tool_call_json_string_format():
    """Verify parsing tool calls from raw JSON string responses (Ollama)."""
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
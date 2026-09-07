import os
import json
import pytest
from unittest.mock import patch, MagicMock
from agent_cli.skill_downloader import (
    ensure_preset_skills_exist,
    find_and_parse_skill_lock,
    has_existing_workspace_skills,
)
from agent_cli.agent_workspace import load_project_skills, ensure_agent_gitignore_entries
from agent_cli.agent_orchestrator import (
    parse_tool_call,
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
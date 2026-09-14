import os
import glob
from rich.console import Console

from agent_workspace_tools import (
    WORKSPACE_TOOL_DISPATCHER,
    execute_async_subprocess,
    get_git_status_changes,
)
from agent_vector_memory.sync import sync_workspace_vector_memory
from agent_vector_memory.store import VectorStoreManager

from .skill_downloader import ensure_preset_skills_exist

console = Console()

AGENT_IGNORES = [
    ".agent/skills/",
    ".skills/",
    ".chroma/",
    ".chromadb/",
    "*.jsonl",
    "agent_traces.jsonl",
    "file_tools_telemetry.jsonl",
    "async_telemetry.jsonl",
    "ollama_debug.log",
    ".codebase_summary.xml",
    "agent_manifest.json"
]


async def initialize_workspace_vector_memory(
    vector_store: VectorStoreManager, 
    workspace_dir: str = "."
) -> None:
    """
    Retrieves uncommitted git changes via agent-workspace-tools and synchronizes
    the vector memory store incrementally (delete-then-upsert indexing).
    """
    abs_workspace = os.path.abspath(workspace_dir)

    # 1. Fetch changed/deleted files using agent-async-runner helper
    is_git_repo, files_to_update, files_to_delete = await get_git_status_changes(abs_workspace)

    # 2. Perform incremental vector store sync
    sync_workspace_vector_memory(
        vector_store=vector_store,
        workspace_dir=abs_workspace,
        is_git_repo=is_git_repo,
        files_to_update=files_to_update,
        files_to_delete=files_to_delete,
    )


async def ensure_agent_gitignore_entries() -> None:
    """Ensures agent runtime files are in .gitignore and commits changes if updated.

    File content flows through WORKSPACE_TOOL_DISPATCHER (read/append) and the
    auto-commit flows through the unified async shell runner.
    """
    if not os.path.exists(os.path.join(os.getcwd(), ".git")):
        return

    gitignore_path = os.path.join(os.getcwd(), ".gitignore")
    header_marker = "# Agent CLI auto-generated artifacts"

    existing_content = ""
    existing_lines: set[str] = set()

    read_file = WORKSPACE_TOOL_DISPATCHER.get("read_file")
    if os.path.exists(gitignore_path) and read_file is not None:
        try:
            res = read_file(file_path=gitignore_path)
            if isinstance(res, dict) and res.get("status") != "ERROR":
                existing_content = res.get("content", "") or ""
                existing_lines = {line.strip() for line in existing_content.splitlines()}
        except Exception:
            pass

    if header_marker in existing_content:
        missing_entries = [entry for entry in AGENT_IGNORES if entry not in existing_lines]
        if not missing_entries:
            return

    to_add = [entry for entry in AGENT_IGNORES if entry not in existing_lines]

    if to_add:
        try:
            prefix = f"\n\n{header_marker}\n" if header_marker not in existing_content else "\n"
            block = prefix + "".join(f"{entry}\n" for entry in to_add)
            appender = WORKSPACE_TOOL_DISPATCHER.get("append_to_file")
            writer = WORKSPACE_TOOL_DISPATCHER.get("write_file")
            if appender is not None and os.path.exists(gitignore_path):
                appender(file_path=gitignore_path, content=block)
            elif writer is not None:
                writer(file_path=gitignore_path, code_body=existing_content + block)
            else:  # pragma: no cover - defensive fallback
                with open(gitignore_path, "a", encoding="utf-8") as f:
                    f.write(block)

            console.print("🛡️  [Git Guard]: Updated [bold].gitignore[/bold] with missing agent artifact patterns.")

            # Commit via the unified workspace shell runner (bypass HITL: internal op).
            staged = await execute_async_subprocess(
                "git add .gitignore", timeout=10.0, bypass_hitl=True
            )
            if staged.get("returncode") == 0:
                await execute_async_subprocess(
                    "git commit -m 'chore: auto-add agent runtime artifacts to .gitignore'",
                    timeout=10.0,
                    bypass_hitl=True,
                )
                console.print("📦 [Git Guard]: Automatically committed .gitignore updates.")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not update/commit .gitignore: {e}[/yellow]")


def load_project_skills() -> str:
    """Scans for skill files in .agent/skills/ and merges them into context.

    File discovery flows through WORKSPACE_TOOL_DISPATCHER (list_files) with a
    glob fallback, and file reads flow through read_file.
    """
    ensure_preset_skills_exist()

    cwd = os.getcwd()
    reader = WORKSPACE_TOOL_DISPATCHER.get("read_file")
    lister = WORKSPACE_TOOL_DISPATCHER.get("list_files")

    # Use absolute path matching to ensure mock directories in tests are resolved
    skill_paths = (
        glob.glob(os.path.join(cwd, ".agent", "skills", "**", "*.[mM][dD]*"), recursive=True) +
        glob.glob(os.path.join(cwd, "skills", "**", "*.[mM][dD]*"), recursive=True)
    )
    if lister is not None:
        for directory in (os.path.join(cwd, ".agent", "skills"), os.path.join(cwd, "skills")):
            try:
                res = lister(directory=directory, recursive=True, pattern="*.md")
                if isinstance(res, dict):
                    for entry in res.get("files", []) or []:
                        p = entry if os.path.isabs(entry) else os.path.join(cwd, entry)
                        if p not in skill_paths:
                            skill_paths.append(p)
            except Exception:
                pass

    if not skill_paths:
        return ""

    skills_text = "\n\nPROJECT SKILLS & DOMAIN GUIDELINES:\n"
    loaded_count = 0

    for path in sorted(set(skill_paths)):
        if os.path.basename(path).startswith("."):
            continue

        try:
            if reader is not None:
                res = reader(file_path=path)
                if not isinstance(res, dict) or res.get("status") == "ERROR":
                    continue
                content = (res.get("content", "") or "").strip()
            else:  # pragma: no cover - defensive fallback
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            rel_path = os.path.relpath(path, start=cwd)
            skills_text += f"\n--- SKILL FILE: {rel_path} ---\n{content}\n"
            loaded_count += 1
        except Exception:
            pass

    if loaded_count > 0:
        console.print(f"🧠 [Skill Loader]: Injected [bold green]{loaded_count}[/bold green] skill file(s) into model context.")
        return skills_text

    return ""
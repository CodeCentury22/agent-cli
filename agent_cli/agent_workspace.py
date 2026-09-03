import os
import glob
import subprocess
from rich.console import Console
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

def ensure_agent_gitignore_entries():
    """Ensures agent runtime files are in .gitignore and commits changes if updated."""
    if not os.path.exists(os.path.join(os.getcwd(), ".git")):
        return

    gitignore_path = os.path.join(os.getcwd(), ".gitignore")
    header_marker = "# Agent CLI auto-generated artifacts"
    
    existing_content = ""
    existing_lines = set()

    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
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
            with open(gitignore_path, "a", encoding="utf-8") as f:
                if header_marker not in existing_content:
                    f.write(f"\n\n{header_marker}\n")
                else:
                    f.write("\n")
                
                for entry in to_add:
                    f.write(f"{entry}\n")
            
            console.print("🛡️  [Git Guard]: Updated [bold].gitignore[/bold] with missing agent artifact patterns.")

            subprocess.run(["git", "add", ".gitignore"], check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "chore: auto-add agent runtime artifacts to .gitignore"],
                check=True,
                capture_output=True
            )
            console.print("📦 [Git Guard]: Automatically committed .gitignore updates.")
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            console.print(f"[yellow]Warning: Could not update/commit .gitignore: {e}[/yellow]")


def load_project_skills() -> str:
    """Scans for skill files in .agent/skills/ and merges them into context."""
    ensure_preset_skills_exist()

    cwd = os.getcwd()
    
    # Use absolute path matching to ensure mock directories in tests are resolved
    skill_paths = (
        glob.glob(os.path.join(cwd, ".agent", "skills", "**", "*.[mM][dD]*"), recursive=True) +
        glob.glob(os.path.join(cwd, "skills", "**", "*.[mM][dD]*"), recursive=True)
    )

    if not skill_paths:
        return ""

    skills_text = "\n\nPROJECT SKILLS & DOMAIN GUIDELINES:\n"
    loaded_count = 0

    for path in sorted(set(skill_paths)):
        if os.path.basename(path).startswith("."):
            continue

        try:
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
import os
import subprocess
import httpx
from rich.console import Console

console = Console()

PRESET_SKILLS = {
    "angular": "https://raw.githubusercontent.com/angular/skills/main/angular-developer/SKILL.md",
    "python": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/python-fastapi.mdc",
    "android": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/android-kotlin.mdc",
    "ios": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/swift-ios.mdc"
}

def ensure_preset_skills_exist():
    """Downloads platform preset skill files ONCE per workspace."""
    skills_dir = os.path.join(os.getcwd(), ".agent", "skills")
    sentinel_file = os.path.join(skills_dir, ".preset_installed")

    if os.path.exists(sentinel_file):
        return

    os.makedirs(skills_dir, exist_ok=True)
    console.print("📥 [Skill Downloader]: Initializing workspace platform skills (one-time setup)...")

    angular_dir = os.path.join(skills_dir, "angular-developer")
    if not os.path.exists(angular_dir):
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/angular/skills.git", angular_dir],
                check=True,
                capture_output=True
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to clone Angular skills: {e}[/yellow]")

    for platform, url in PRESET_SKILLS.items():
        if platform == "angular":
            continue

        file_path = os.path.join(skills_dir, f"{platform}.md")
        if not os.path.exists(file_path):
            try:
                with httpx.Client(follow_redirects=True, timeout=5.0) as client:
                    res = client.get(url)
                    if res.status_code == 200:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(res.text.strip())
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to download {platform} skill: {e}[/yellow]")

    try:
        with open(sentinel_file, "w", encoding="utf-8") as f:
            f.write("installed")
    except Exception:
        pass
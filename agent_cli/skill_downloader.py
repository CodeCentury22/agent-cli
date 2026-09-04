import os
import subprocess
import httpx
from rich.console import Console

console = Console()

PRESET_SKILLS = {
    "angular": "https://raw.githubusercontent.com/angular/skills/main/angular-developer/SKILL.md",
    "python": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/python-fastapi-best-practices-cursorrules-prompt-f.mdc",
    "csharp": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/unity-cursor-ai-c-cursorrules-prompt-file.mdc",
    "cpp": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/cpp.mdc",
    "react": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/react.mdc",
    "vue": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/vue.mdc",
    "rust": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/rust.mdc",
    "go": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/go.mdc",
    "android": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/android-jetpack-compose-cursorrules-prompt-file.mdc",
    "ios": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/swiftui-guidelines-cursorrules-prompt-file.mdc",
    "java": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/java-springboot-jpa-cursorrules-prompt-file.mdc",
    "flutter": "https://raw.githubusercontent.com/PatrickJS/awesome-cursorrules/main/rules/flutter-development-guidelines-cursorrules-prompt-file.mdc",
}

def detect_project_platforms() -> list[str]:
    """Detects active frameworks/languages in the current workspace."""
    cwd = os.getcwd()
    detected = []

    # Angular
    if os.path.exists(os.path.join(cwd, "angular.json")):
        detected.append("angular")

    # JS/TS Frameworks
    pkg_json_path = os.path.join(cwd, "package.json")
    if os.path.exists(pkg_json_path):
        try:
            with open(pkg_json_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "react" in content:
                    detected.append("react")
                if "vue" in content:
                    detected.append("vue")
        except Exception:
            pass

    # Python
    if any(os.path.exists(os.path.join(cwd, f)) for f in ["pyproject.toml", "requirements.txt", "Pipfile", "setup.py"]):
        detected.append("python")

    # Java
    if os.path.exists(os.path.join(cwd, "pom.xml")) or os.path.exists(os.path.join(cwd, "build.gradle")):
        # Only classify as general java if not specifically Android
        if not os.path.exists(os.path.join(cwd, "app", "build.gradle")):
            detected.append("java")

    # Flutter / Dart
    if os.path.exists(os.path.join(cwd, "pubspec.yaml")):
        detected.append("flutter")

    # C# / .NET
    if any(f.endswith((".csproj", ".sln")) for f in os.listdir(cwd)):
        detected.append("csharp")

    # C++
    if os.path.exists(os.path.join(cwd, "CMakeLists.txt")) or any(f.endswith((".cpp", ".hpp", ".cc", ".cxx")) for f in os.listdir(cwd)):
        detected.append("cpp")

    # Rust
    if os.path.exists(os.path.join(cwd, "Cargo.toml")):
        detected.append("rust")

    # Go
    if os.path.exists(os.path.join(cwd, "go.mod")):
        detected.append("go")

    # Android
    if os.path.exists(os.path.join(cwd, "build.gradle")) or os.path.exists(os.path.join(cwd, "build.gradle.kts")):
        detected.append("android")

    # iOS
    if any(f.endswith((".xcodeproj", ".xcworkspace")) for f in os.listdir(cwd)):
        detected.append("ios")

    return list(set(detected))


def ensure_preset_skills_exist():
    """Detects workspace stack and downloads matching platform skills ONCE."""
    skills_dir = os.path.join(os.getcwd(), ".agent", "skills")
    sentinel_file = os.path.join(skills_dir, ".preset_installed")

    if os.path.exists(sentinel_file):
        return

    platforms = detect_project_platforms()
    if not platforms:
        return

    os.makedirs(skills_dir, exist_ok=True)
    console.print(f"📥 [Skill Downloader]: Detected stack: [cyan]{', '.join(platforms)}[/cyan]. Downloading preset skills...")

    for platform in platforms:
        if platform == "angular":
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
        elif platform in PRESET_SKILLS:
            url = PRESET_SKILLS[platform]
            file_path = os.path.join(skills_dir, f"{platform}.md")
            if not os.path.exists(file_path):
                try:
                    with httpx.Client(follow_redirects=True, timeout=10.0) as client:
                        res = client.get(url)
                        if res.status_code == 200:
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(res.text.strip())
                        else:
                            console.print(f"[yellow]Warning: Skill download for '{platform}' returned HTTP {res.status_code}[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to download {platform} skill: {e}[/yellow]")
        else:
            console.print(f"[dim]Note: Platform '{platform}' detected, but no matching preset skill URL configured.[/dim]")

    try:
        with open(sentinel_file, "w", encoding="utf-8") as f:
            f.write("installed")
    except Exception:
        pass
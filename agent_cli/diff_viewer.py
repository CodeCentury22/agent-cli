import difflib
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()

def render_file_diff(file_path: str, old_content: str, new_content: str):
    """Renders a colorful unified diff for file modifications similar to Cline."""
    diff_lines = list(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )

    if not diff_lines:
        console.print(f"[dim]No changes detected for {file_path}[/dim]")
        return

    diff_text = "".join(diff_lines)
    syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
    
    console.print(Panel(
        syntax,
        title=f"📝 [bold yellow]Proposed Changes: {file_path}[/bold yellow]",
        border_style="cyan",
        expand=False
    ))
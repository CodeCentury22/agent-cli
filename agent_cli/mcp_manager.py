import os
import json
from rich.console import Console
from rich.panel import Panel

console = Console()

AGENT_DIR = ".agent"
MCP_CONFIG_PATH = os.path.join(AGENT_DIR, "mcp.json")

INITIAL_MCP_TEMPLATE = {
    "mcpServers": {
        "// example_server": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-git"]
        }
    }
}


def display_mcp_guidance():
    """Prints terminal panel reminding the developer how to add MCP servers."""
    console.print(
        Panel(
            f"[bold cyan]🔌 Model Context Protocol (MCP) Available[/bold cyan]\n\n"
            f"Configuration file location: [bold yellow]{MCP_CONFIG_PATH}[/bold yellow]\n\n"
            f"[white]To attach external tool servers (Git, Postgres, Angular, etc.), add your server definitions inside:[/white]\n"
            f"[bold green].agent/mcp.json[/bold green]\n\n"
            f"[dim]Expected Format:\n"
            f"{{\n"
            f'  "mcpServers": {{\n'
            f'    "server-name": {{\n'
            f'      "command": "npx|uvx|node|python",\n'
            f'      "args": ["--flag", "val"]\n'
            f"    }}\n"
            f"  }}\n"
            f"}}[/dim]",
            title="MCP Guidance",
            border_style="cyan"
        )
    )


def ensure_and_load_mcp_servers() -> dict:
    """
    Ensures .agent/mcp.json exists. Retains all local skill files.
    If no active servers are defined, displays guidance on every run.
    """
    os.makedirs(AGENT_DIR, exist_ok=True)

    # 1. Ensure boilerplate template exists
    if not os.path.exists(MCP_CONFIG_PATH):
        try:
            with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(INITIAL_MCP_TEMPLATE, f, indent=2)
        except Exception as e:
            console.print(f"⚠️ [MCP Manager]: Failed to write default config to {MCP_CONFIG_PATH}: {e}")

    # 2. Parse active servers (skipping comment keys starting with '//')
    active_servers = {}
    if os.path.exists(MCP_CONFIG_PATH):
        try:
            with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_servers = data.get("mcpServers") or data.get("servers") or {}
                active_servers = {
                    name: cfg for name, cfg in raw_servers.items() if not name.startswith("//")
                }
        except Exception as e:
            console.print(f"⚠️ [MCP Manager]: Error reading {MCP_CONFIG_PATH}: {e}")

    # 3. Inform user without touching skill files
    if not active_servers:
        display_mcp_guidance()
    else:
        console.print(f"🔌 [MCP Manager]: Loaded [bold green]{len(active_servers)}[/bold green] configured MCP server(s).")

    return active_servers
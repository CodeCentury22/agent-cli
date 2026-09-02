import os
import sys
import subprocess
import json
import glob
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from typing import Tuple
from .diff_viewer import render_file_diff

from agent_llm_client import create_llm_client, BaseLLMClient, OllamaClient
from agent_vector_memory import VectorStoreManager
from .auth import get_stored_credentials, save_credentials, start_browser_oauth_flow

from agent_file_tools import FILE_TOOLS_SCHEMA, TOOL_DISPATCHER
from agent_async_runner import SHELL_TOOLS_SCHEMA, ASYNC_TOOL_DISPATCHER
from agent_guardrails import validate_tool_args


# Combine all tool schemas for Ollama / Gemini / Claude / OpenRouter
ALL_TOOLS_SCHEMA = FILE_TOOLS_SCHEMA + SHELL_TOOLS_SCHEMA

# Combine all dispatchers
ALL_TOOL_DISPATCHERS = {**TOOL_DISPATCHER, **ASYNC_TOOL_DISPATCHER}

console = Console()

PROVIDERS = {
    "1": {
        "name": "ollama",
        "label": "Local (Ollama)",
        "models": [
            "qwen2.5-coder:7b-instruct",
            "qwen2.5-coder:32b-instruct",
            "deepseek-r1:32b",
            "llama3.3:70b",
            "deepseek-r1:70b"
        ]
    },
    "2": {
        "name": "gemini",
        "label": "Google Gemini",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"]
    },
    "3": {
        "name": "claude",
        "label": "Anthropic Claude",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
    },
    "4": {
        "name": "openrouter",
        "label": "OpenRouter / OpenCode Gateway",
        "models": [
            "deepseek/deepseek-r1",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.3-70b-instruct"
        ]
    }
}

AGENT_IGNORES = [
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


def load_project_skills() -> str:
    """Scans for skill Markdown files in .agent/skills/ or skills/ and merges them into context."""
    skill_paths = glob.glob(".agent/skills/*.md") + glob.glob("skills/*.md") + glob.glob(".skills/*.md")
    
    if not skill_paths:
        return ""

    skills_text = "\n\nPROJECT SKILLS & DOMAIN GUIDELINES:\n"
    loaded_count = 0

    for path in skill_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                skills_text += f"\n--- SKILL: {os.path.basename(path)} ---\n{content}\n"
                loaded_count += 1
        except Exception:
            pass

    if loaded_count > 0:
        console.print(f"🧠 [Skill Loader]: Injected [bold green]{loaded_count}[/bold green] skill guide(s) into model memory.")
        return skills_text

    return ""

def ensure_agent_gitignore_entries():
    """Ensures agent runtime files are in .gitignore and commits changes if updated."""
    # Skip if not inside a git repository
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

    # Quick check: If header marker is already present and all items exist, exit early
    if header_marker in existing_content:
        missing_entries = [entry for entry in AGENT_IGNORES if entry not in existing_lines]
        if not missing_entries:
            return  # Clean exit, nothing to append or commit

    to_add = [entry for entry in AGENT_IGNORES if entry not in existing_lines]

    if to_add:
        try:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                # Write header marker only if not present yet
                if header_marker not in existing_content:
                    f.write(f"\n\n{header_marker}\n")
                else:
                    f.write("\n")
                
                for entry in to_add:
                    f.write(f"{entry}\n")
            
            console.print("🛡️  [Git Guard]: Updated [bold].gitignore[/bold] with missing agent artifact patterns.")

            # Stage and commit the .gitignore update
            subprocess.run(["git", "add", ".gitignore"], check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "chore: auto-add agent runtime artifacts to .gitignore"],
                check=True,
                capture_output=True
            )
            console.print("📦 [Git Guard]: Automatically committed .gitignore updates.")
        except subprocess.CalledProcessError:
            # Handle cases where git working tree has no changes to commit
            pass
        except Exception as e:
            console.print(f"[yellow]Warning: Could not update/commit .gitignore: {e}[/yellow]")


async def handle_tool_call(tool_name: str, raw_args: dict):
    """Sanitizes arguments via guardrails, displays diffs for file edits, and dispatches tools."""
    is_valid, sanitized_args, error_msg = validate_tool_args(tool_name, raw_args)
    
    if not is_valid:
        console.print(f"❌ [Guardrail Reject]: {error_msg}")
        return {"status": "ERROR", "error": error_msg}

    if tool_name == "write_file":
        target_path = sanitized_args.get("file_path")
        new_code = sanitized_args.get("code_body", "")
        old_code = ""

        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    old_code = f.read()
            except Exception:
                old_code = ""

        render_file_diff(target_path, old_code, new_code)

    dispatcher = ALL_TOOL_DISPATCHERS.get(tool_name)
    if not dispatcher:
        return {"status": "ERROR", "error": f"Tool '{tool_name}' not registered."}

    # Execute dispatcher and inspect result for coroutines
    result = dispatcher(**sanitized_args)
    if asyncio.iscoroutine(result):
        result = await result

    return result


def setup_provider_and_auth() -> Tuple[str, str, str | None]:
    """Interactive wizard for provider selection, model choice, and auth key handling."""
    console.print(Panel("[bold cyan]🤖 Agent CLI - Interactive Workspace Session[/bold cyan]", expand=False))

    console.print("\n[bold yellow]Select LLM Provider:[/bold yellow]")
    for key, info in PROVIDERS.items():
        console.print(f"    [bold green]{key})[/bold green] {info['label']}")

    provider_choice = Prompt.ask("\nChoose provider", choices=list(PROVIDERS.keys()), default="1")
    selected_provider = PROVIDERS[provider_choice]["name"]
    available_models = PROVIDERS[provider_choice]["models"]

    console.print(f"\n[bold yellow]Select Model for {PROVIDERS[provider_choice]['label']}:[/bold yellow]")
    for idx, model in enumerate(available_models, 1):
        console.print(f"    [bold green]{idx})[/bold green] {model}")

    model_choice = Prompt.ask(
        "Choose model",
        choices=[str(i) for i in range(1, len(available_models) + 1)],
        default="1"
    )
    selected_model = available_models[int(model_choice) - 1]

    api_key = None
    if selected_provider != "ollama":
        api_key = get_stored_credentials(selected_provider)
        if api_key:
            console.print(f"🔑 Using stored credentials for [bold]{selected_provider}[/bold].")
        else:
            if selected_provider == "gemini":
                auth_mode = Prompt.ask("Auth method", choices=["api_key", "browser"], default="api_key")
            else:
                auth_mode = "api_key"
            if auth_mode == "browser":
                oauth_url = "https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id=YOUR_CLIENT_ID"
                api_key = start_browser_oauth_flow(oauth_url)
            else:
                api_key = Prompt.ask(f"Enter API Key for [bold]{selected_provider}[/bold]", password=True)

            if api_key:
                save_credentials(selected_provider, api_key)
    return selected_provider, selected_model, api_key

async def async_main():
    try:
        ensure_agent_gitignore_entries()
        provider, model, api_key = setup_provider_and_auth()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Session canceled.[/yellow]")
        sys.exit(0)
        
    console.print(f"\n🚀 Initializing LLM Client [[bold cyan]{provider}[/bold cyan] :: [bold green]{model}[/bold green]]...")

    try:
        llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key
        )
        if isinstance(llm_client, OllamaClient):
            if not await llm_client.ensure_model_available():
                sys.exit(1)
        vector_store = VectorStoreManager(llm_client=llm_client)
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Session Active![/bold]\n"
        f"Directory: [cyan]{os.getcwd()}[/cyan]\n"
        f"Provider:  [green]{provider}[/green]\n"
        f"Model:     [yellow]{model}[/yellow]\n\n"
        f"Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to end the session.",
        title="Agent Environment"
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]agent>[/bold green]").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Ending session. Goodbye![/yellow]")
                break
                
            context_matches = vector_store.search_codebase(user_input, top_k=2)
            context_str = "\n".join([f"File: {m['file_path']}\nContent: {m['content']}" for m in context_matches])
            skills_context = load_project_skills()
            system_prompt = (
                "You are an autonomous software engineering agent operating in a CLI workspace.\n\n"
                "OPERATIONAL RULES:\n"
                "1. Always inspect configuration files (e.g., `package.json`, build configs) before executing shell commands.\n"
                "2. When calling tools, output valid JSON strictly matching the schema:\n"
                '   {"name": "tool_name", "arguments": {"arg": "value"}}\n'
                "3. If a tool command or build fails, DO NOT repeat identical arguments. Read error output, inspect files, or adjust flags.\n"
                "4. Adhere strictly to project-specific skills and domain guidelines if they areprovided below.\n"
                f"{skills_context}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_str}\n\nTask: {user_input}"}
            ]

            last_tool_signature = None

            # Multi-turn tool execution loop
            while True:
                # Prune old context messages to avoid token bloat during deep turns
                if len(messages) > 12:
                    messages = [messages[0], messages[1]] + messages[-8:]

                response_obj, metrics = await llm_client.chat(messages, tools=ALL_TOOLS_SCHEMA)

                tool_name = None
                raw_args = {}

                # 1. Check object attribute format (SDK models)
                if hasattr(response_obj, "tool_calls") and response_obj.tool_calls:
                    call = response_obj.tool_calls[0]
                    tool_name = call.get("name")
                    raw_args = call.get("arguments", {})
                
                # 2. Check raw JSON string format (Ollama / JSON-formatted providers)
                elif isinstance(response_obj, str):
                    try:
                        parsed = json.loads(response_obj)
                        if isinstance(parsed, dict):
                            tool_name = parsed.get("name") or parsed.get("tool_name")
                            raw_args = parsed.get("arguments", {})
                    except (json.JSONDecodeError, TypeError):
                        pass

                # If a valid tool execution was requested:
                if tool_name and tool_name in ALL_TOOL_DISPATCHERS:
                    tool_signature = (tool_name, json.dumps(raw_args, sort_keys=True))
                    
                    # Circuit breaker: catch repeated identical calls
                    if tool_signature == last_tool_signature:
                        console.print(f"\n🛑 [Circuit Breaker]: Detected duplicate call to '{tool_name}'. Halting turn.")
                        messages.append({
                            "role": "user",
                            "content": f"System Warning: Do not repeat failed command '{tool_name}' with identical arguments."
                        })
                        last_tool_signature = None
                        break

                    last_tool_signature = tool_signature

                    console.print(f"\n🛠️  [bold yellow]Agent Invoking Tool:[/bold yellow] [cyan]{tool_name}[/cyan]")
                    
                    tool_result = await handle_tool_call(tool_name, raw_args)
                    console.print(f"📋 [bold green]Tool Execution Result:[/bold green]\n{tool_result}")

                    # Append tool invocation and output to context history
                    messages.append({"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": raw_args})})
                    messages.append({
                        "role": "user",
                        "content": f"Tool '{tool_name}' Output:\n{tool_result}\n\nContinue with task or call next tool."
                    })

                    console.print(f"[dim]Metrics: {metrics}[/dim]")
                    # Loop again to pass the tool output back to the model
                    continue

                # Final response reached (no further tool calls requested)
                console.print(f"\n🤖 [bold cyan]Agent Response:[/bold cyan]\n{response_obj}")
                console.print(f"\n[dim]Metrics: {metrics}[/dim]")
                break

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
            break

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
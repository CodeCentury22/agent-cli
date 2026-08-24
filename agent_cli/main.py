import os
import sys
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from typing import Tuple

from agent_llm_client import create_llm_client, BaseLLMClient
from agent_vector_memory import VectorStoreManager
from .auth import get_stored_credentials, save_credentials, start_browser_oauth_flow

from agent_file_tools import FILE_TOOLS_SCHEMA, TOOL_DISPATCHER
from agent_async_runner import SHELL_TOOLS_SCHEMA, ASYNC_TOOL_DISPATCHER
from agent_guardrails import validate_tool_args


# Combine all tool schemas for Ollama / Gemini
ALL_TOOLS_SCHEMA = FILE_TOOLS_SCHEMA + SHELL_TOOLS_SCHEMA

# Combine all dispatchers
ALL_TOOL_DISPATCHERS = {**TOOL_DISPATCHER, **ASYNC_TOOL_DISPATCHER}

console = Console()

PROVIDERS = {
    "1": {
        "name": "ollama",
        "label": "Local (Ollama)",
        "models": ["qwen2.5-coder:7b-instruct", "llama3.3:70b", "deepseek-r1:70b"]
    },
    "2": {
        "name": "gemini",
        "label": "Google Gemini",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"]
    },
    "3": {
        "name": "claude",  # Fixed lowercased provider key
        "label": "Anthropic Claude",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
    },
    "4": {
        "name": "openrouter",
        "label": "OpenRouter / OpenCode Gateway",
        "models": ["deepseek/deepseek-r1", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.3-70b-instruct"]
    }
}

async def handle_tool_call(tool_name: str, raw_args: dict):
    """Sanitizes arguments via guardrails and executes the target tool dispatcher."""
    # 1. Parameter Guardrail Validation & Path Normalization
    is_valid, sanitized_args, error_msg = validate_tool_args(tool_name, raw_args)
    
    if not is_valid:
        print(f"❌ [Guardrail Reject]: {error_msg}")
        return {"status": "ERROR", "error": error_msg}

    # 2. Dispatch Tool Call
    dispatcher = ALL_TOOL_DISPATCHERS.get(tool_name)
    if not dispatcher:
        return {"status": "ERROR", "error": f"Tool '{tool_name}' not registered."}

    # 3. Handle Async vs Sync execution transparently
    if asyncio.iscoroutinefunction(dispatcher):
        return await dispatcher(**sanitized_args)
    else:
        return dispatcher(**sanitized_args)

def setup_provider_and_auth() -> Tuple[str, str, str | None]:
    """Interactive wizard for provider selection, model choice, and auth key handling."""
    console.print(Panel("[bold cyan]🤖 Agent CLI - Interactive Workspace Session[/bold cyan]", expand=False))

    console.print("\n[bold yellow]Select LLM Provider:[/bold yellow]")  # Fixed markup typo
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

            messages = [
                {"role": "system", "content": "You are a software engineering assistant operating in a workspace CLI."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nTask: {user_input}"}
            ]

            # 1. Pass ALL_TOOLS_SCHEMA into the LLM chat call
            response_obj, metrics = await llm_client.chat(messages, tools=ALL_TOOLS_SCHEMA)

            # 2. Check if the LLM returned tool calls to execute
            # 2. Check if the LLM returned tool calls to execute
            if hasattr(response_obj, "tool_calls") and response_obj.tool_calls:
                for tool_call in response_obj.tool_calls:
                    tool_name = tool_call["name"]
                    raw_args = tool_call["arguments"]
                    tool_call_id = tool_call.get("id", "call_default")

                    console.print(f"\n🛠️  [bold yellow]Agent Invoking Tool:[/bold yellow] [cyan]{tool_name}[/cyan]")
                    
                    # 3. Route through guardrails, HITL, and dispatchers
                    tool_result = await handle_tool_call(tool_name, raw_args)
                    console.print(f"📋 [bold green]Tool Execution Result:[/bold green]\n{tool_result}")

                    # Append message history with matching tool_call_id
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": str(tool_result)
                    })
                
                # Re-query the model with tool execution feedback so it can generate its final answer
                final_response, _ = await llm_client.chat(messages)
                console.print(f"\n🤖 [bold cyan]Agent Response:[/bold cyan]\n{final_response}")
            else:
                # Direct text response (no tool execution requested)
                console.print(f"\n🤖 [bold cyan]Agent Response:[/bold cyan]\n{response_obj}")
            
            console.print(f"\n[dim]Metrics: {metrics}[/dim]")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
            break

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
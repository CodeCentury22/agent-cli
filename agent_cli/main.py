import os
import sys
import asyncio
import importlib.metadata
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from agent_llm_client import create_llm_client, BaseLLMClient, OllamaClient
from agent_vector_memory import VectorStoreManager

from .agent_workspace import ensure_agent_gitignore_entries
from .agent_config import setup_provider_and_auth
from .agent_orchestrator import run_agent_turn
from .skill_downloader import ensure_preset_skills_exist

console = Console()

try:
    VERSION = f"v{importlib.metadata.version('agent-cli')}"
except importlib.metadata.PackageNotFoundError:
    VERSION = "v0.6.1"

def display_welcome_banner():
    console.print(
        Panel(
            f"🤖 [bold cyan]Agent CLI[/bold cyan] [dim]({VERSION})[/dim] - Interactive Workspace Session",
            expand=False,
            border_style="cyan"
        )
    )

async def async_main():
    display_welcome_banner()
    
    # Ensure gitignore rules and workspace preset skills exist
    ensure_agent_gitignore_entries()
    ensure_preset_skills_exist()

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
        if isinstance(llm_client, OllamaClient):
            if not await llm_client.ensure_model_available():
                sys.exit(1)
        
        vector_store = VectorStoreManager(llm_client=llm_client)
        
        # Index workspace files into vector memory
        with console.status("[bold cyan]Indexing workspace into vector memory...[/bold cyan]"):
            if asyncio.iscoroutinefunction(getattr(vector_store, "index_workspace", None)):
                await vector_store.index_workspace()
            else:
                vector_store.index_workspace()
                
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

    # Configure prompt_toolkit style and async PromptSession
    prompt_style = Style.from_dict({
        'prompt': 'ansigreen bold',
    })
    session = PromptSession(style=prompt_style)

    while True:
        try:
            console.print()  # Add spacing before prompt
            # Use await session.prompt_async(...) inside active asyncio loop
            user_input = (await session.prompt_async(HTML('<prompt>agent&gt; </prompt>'))).strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Ending session. Goodbye![/yellow]")
                break
                
            await run_agent_turn(user_input, llm_client, vector_store)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
            break

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
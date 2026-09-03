import os
import sys
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from agent_llm_client import create_llm_client, BaseLLMClient, OllamaClient
from agent_vector_memory import VectorStoreManager

from .agent_workspace import ensure_agent_gitignore_entries
from .agent_config import setup_provider_and_auth
from .agent_orchestrator import run_agent_turn

console = Console()

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
                
            await run_agent_turn(user_input, llm_client, vector_store)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
            break

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
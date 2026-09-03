from typing import Tuple
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .auth import get_stored_credentials, save_credentials, start_browser_oauth_flow

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
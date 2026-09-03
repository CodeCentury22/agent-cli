import json
from rich.console import Console
from agent_llm_client import BaseLLMClient
from agent_vector_memory import VectorStoreManager
from agent_file_tools import FILE_TOOLS_SCHEMA, TOOL_DISPATCHER
from agent_async_runner import SHELL_TOOLS_SCHEMA, ASYNC_TOOL_DISPATCHER
from .tool_handler import handle_tool_call
from .agent_workspace import load_project_skills

ALL_TOOLS_SCHEMA = FILE_TOOLS_SCHEMA + SHELL_TOOLS_SCHEMA
ALL_TOOL_DISPATCHERS = {**TOOL_DISPATCHER, **ASYNC_TOOL_DISPATCHER}

console = Console()

def parse_tool_call(response_obj) -> tuple[str | None, dict]:
    """Extracts tool_name and raw_args from SDK object or JSON response."""
    if hasattr(response_obj, "tool_calls") and response_obj.tool_calls:
        call = response_obj.tool_calls[0]
        return call.get("name"), call.get("arguments", {})
    elif isinstance(response_obj, str):
        try:
            parsed = json.loads(response_obj)
            if isinstance(parsed, dict):
                return parsed.get("name") or parsed.get("tool_name"), parsed.get("arguments", {})
        except (json.JSONDecodeError, TypeError):
            pass
    return None, {}

async def run_agent_turn(user_input: str, llm_client: BaseLLMClient, vector_store: VectorStoreManager):
    """Executes a complete single-user-request turn with multi-turn tool calling."""
    context_matches = vector_store.search_codebase(user_input, top_k=2)
    context_str = "\n".join([f"File: {m['file_path']}\nContent: {m['content']}" for m in context_matches])
    
    # Dynamic re-scan of installed skills
    skills_context = load_project_skills()

    system_prompt = (
        "You are an autonomous software engineering agent operating in a CLI workspace.\n\n"
        "OPERATIONAL RULES:\n"
        "1. Always inspect configuration files (e.g., `package.json`, build configs) before executing shell commands.\n"
        "2. Determine project platform and adhere strictly to matching guidelines in injected PROJECT SKILLS & DOMAIN GUIDELINES.\n"
        "3. When calling tools, output valid JSON strictly matching the schema:\n"
        '   {"name": "tool_name", "arguments": {"arg": "value"}}\n'
        "4. If a tool command or build fails, DO NOT repeat identical arguments. Read error output, inspect files, or adjust flags.\n"
        f"{skills_context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nTask: {user_input}"}
    ]

    last_tool_signature = None

    while True:
        # Prune old context messages to avoid token bloat during deep turns
        if len(messages) > 12:
            messages = [messages[0], messages[1]] + messages[-8:]

        response_obj, metrics = await llm_client.chat(messages, tools=ALL_TOOLS_SCHEMA)
        tool_name, raw_args = parse_tool_call(response_obj)

        if tool_name and tool_name in ALL_TOOL_DISPATCHERS:
            tool_signature = (tool_name, json.dumps(raw_args, sort_keys=True))
            
            # Circuit breaker
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
            
            tool_result = await handle_tool_call(tool_name, raw_args, ALL_TOOL_DISPATCHERS)
            console.print(f"📋 [bold green]Tool Execution Result:[/bold green]\n{tool_result}")

            messages.append({"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": raw_args})})
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' Output:\n{tool_result}\n\nContinue with task or call next tool."
            })

            console.print(f"[dim]Metrics: {metrics}[/dim]")
            continue

        console.print(f"\n🤖 [bold cyan]Agent Response:[/bold cyan]\n{response_obj}")
        console.print(f"\n[dim]Metrics: {metrics}[/dim]")
        break
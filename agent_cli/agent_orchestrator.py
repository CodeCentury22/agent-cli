import os
import re
import json
from rich.console import Console
from agent_llm_client import BaseLLMClient
from agent_vector_memory import VectorStoreManager
from agent_file_tools import FILE_TOOLS_SCHEMA, TOOL_DISPATCHER
from agent_guardrails import validate_tool_args
from agent_async_runner import SHELL_TOOLS_SCHEMA, ASYNC_TOOL_DISPATCHER
from .tool_handler import handle_tool_call
from .agent_workspace import load_project_skills

ALL_TOOLS_SCHEMA = FILE_TOOLS_SCHEMA + SHELL_TOOLS_SCHEMA
ALL_TOOL_DISPATCHERS = {**TOOL_DISPATCHER, **ASYNC_TOOL_DISPATCHER}

console = Console()


def load_workspace_config_context(workspace_dir: str = ".") -> str:
    """Reads project configuration files to ground the system prompt context."""
    configs = ["package.json", "angular.json", "tsconfig.json"]
    parts = []
    for cfg in configs:
        cfg_path = os.path.join(workspace_dir, cfg)
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    parts.append(f"--- {cfg} ---\n{content[:1200]}")
            except Exception:
                pass
    return "\n".join(parts)


def parse_tool_call(response_obj) -> tuple[str | None, dict]:
    """Extracts tool_name and raw_args from SDK object, raw JSON, or markdown-wrapped responses."""
    # 1. Native SDK Tool Call Objects
    if hasattr(response_obj, "tool_calls") and response_obj.tool_calls:
        call = response_obj.tool_calls[0]
        return call.get("name"), call.get("arguments", {})

    elif isinstance(response_obj, str):
        # 2. Pure JSON String
        try:
            parsed = json.loads(response_obj)
            if isinstance(parsed, dict):
                name = parsed.get("name") or parsed.get("tool_name")
                args = parsed.get("arguments") or parsed.get("args") or {}
                if name:
                    return name, args
        except (json.JSONDecodeError, TypeError):
            pass

        # 3. Extract JSON embedded in markdown code fences (```json ... ``` or ``` ...)
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_obj, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    name = parsed.get("name") or parsed.get("tool_name")
                    args = parsed.get("arguments") or parsed.get("args") or {}
                    if name:
                        return name, args
            except (json.JSONDecodeError, TypeError):
                pass

        # 4. Fallback: Search for any JSON dict structure containing "name" or "tool_name"
        match = re.search(r"(\{\s*\"(?:name|tool_name)\"[\s\S]*\})", response_obj)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    name = parsed.get("name") or parsed.get("tool_name")
                    args = parsed.get("arguments") or parsed.get("args") or {}
                    if name:
                        return name, args
            except (json.JSONDecodeError, TypeError):
                pass

    return None, {}


async def run_agent_turn(user_input: str, llm_client: BaseLLMClient, vector_store: VectorStoreManager):
    """Executes a complete single-user-request turn with multi-turn tool calling and guardrails."""
    context_matches = vector_store.search_codebase(user_input, top_k=3)
    context_str = "\n".join([f"File: {m['file_path']}\nContent: {m['content']}" for m in context_matches])
    
    # Load dynamic skill files and project root configurations
    skills_context = load_project_skills()
    config_context = load_workspace_config_context()

    system_prompt = (
        "You are an autonomous software engineering agent operating in a CLI workspace.\n\n"
        "VECTOR CONTEXT & FILE DISCOVERY RULES:\n"
        "1. You MUST strictly rely on the provided Chroma vector search results injected into your context to locate files and understand project components.\n"
        "2. DO NOT execute terminal shell commands (e.g., `find`, `grep`, `locate`, `ls -R`) to discover or search for files.\n"
        "3. BEFORE modifying or creating any files, verify that the target component exists within the provided vector results or project configurations. "
        "If the required file or component is NOT present in the vector context, TERMINATE the session immediately and inform the user.\n"
        "4. DO NOT create duplicate folders or component scaffolding if the vector context yields no exact match.\n\n"
        "COMMAND EXECUTION & SAFETY RULES:\n"
        "1. NEVER execute interactive daemons or MCP servers in foreground turns (e.g., `ng mcp`, `ng serve`). "
        "Only run non-blocking CLI commands, or spawn background jobs for builds/tests using `start_background_task` and inspect them using `get_background_task_status`.\n"
        "2. Always inspect configuration files (e.g., `package.json`, build configs) before executing shell commands.\n"
        "3. Determine project platform and adhere strictly to matching guidelines in injected PROJECT SKILLS & DOMAIN GUIDELINES.\n"
        "4. When calling tools, output valid JSON strictly matching the schema:\n"
        '   {"name": "tool_name", "arguments": {"arg": "value"}}\n'
        "5. If a tool command or build fails, DO NOT repeat identical arguments. Read error output, inspect files, or adjust flags.\n\n"
        f"WORKSPACE CONFIGURATIONS:\n{config_context}\n\n"
        f"{skills_context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nTask: {user_input}"}
    ]

    recent_tool_signatures = []

    while True:
        # Prune old context messages to avoid token bloat during deep turns
        if len(messages) > 12:
            messages = [messages[0], messages[1]] + messages[-8:]

        response_obj, metrics = await llm_client.chat(messages, tools=ALL_TOOLS_SCHEMA)
        tool_name, raw_args = parse_tool_call(response_obj)

        if tool_name and tool_name in ALL_TOOL_DISPATCHERS:
            # Pre-validate arguments through Pydantic self-healing schemas (e.g. path -> file_path)
            is_valid, validated_args, err_msg = validate_tool_args(tool_name, raw_args)
            
            if not is_valid:
                console.print(f"\n⚠️ [Schema Validation Error]: {err_msg}. Triggering prompt repair...")
                messages.append({"role": "assistant", "content": str(response_obj)})
                messages.append({
                    "role": "user",
                    "content": f"System Error: Tool call '{tool_name}' arguments were invalid: {err_msg}. Please fix the arguments according to the schema and retry."
                })
                continue

            tool_signature = (tool_name, json.dumps(validated_args, sort_keys=True))
            
            # Sliding window circuit breaker (catches alternating loops like which ng -> ng version)
            if recent_tool_signatures.count(tool_signature) >= 2:
                console.print(f"\n🛑 [Circuit Breaker]: Detected repeating tool call loop for '{tool_name}'. Halting turn.")
                messages.append({
                    "role": "user",
                    "content": f"System Warning: Stop repeating command '{tool_name}'. Proceed to next task step or return status."
                })
                break

            recent_tool_signatures.append(tool_signature)
            if len(recent_tool_signatures) > 6:
                recent_tool_signatures.pop(0)

            console.print(f"\n🛠️  [bold yellow]Agent Invoking Tool:[/bold yellow] [cyan]{tool_name}[/cyan]")
            
            tool_result = await handle_tool_call(tool_name, validated_args, ALL_TOOL_DISPATCHERS)
            console.print(f"📋 [bold green]Tool Execution Result:[/bold green]\n{tool_result}")

            messages.append({"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": validated_args})})
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' Output:\n{tool_result}\n\nContinue with task or call next tool."
            })

            console.print(f"[dim]Metrics: {metrics}[/dim]")
            continue

        console.print(f"\n🤖 [bold cyan]Agent Response:[/bold cyan]\n{response_obj}")
        console.print(f"\n[dim]Metrics: {metrics}[/dim]")
        break
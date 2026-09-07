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
from .mcp_manager import ensure_and_load_mcp_servers, get_mcp_tool_schemas_and_dispatchers

console = Console()

README_CANDIDATES = [
    "README.md",
    "README.txt",
    "README",
    "readme.md",
    "docs/README.md",
]


def sanitize_readme_for_llm(readme_text: str, max_chars: int = 3500) -> str:
    """Strips visual noise, HTML comments, and marketing images from README prior to prompt injection."""
    if not readme_text:
        return ""

    # 1. Strip Markdown images and HTML badges
    text = re.sub(r"!\[.*?\]\(.*?\)", "", readme_text)
    text = re.sub(r"<img.*?>", "", text, flags=re.IGNORECASE)

    # 2. Strip HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 3. Truncate excess length while preserving operational sections
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[README truncated for context efficiency]..."

    return text.strip()


def read_workspace_readme(workspace_dir: str = ".") -> str:
    """Locates, reads, and sanitizes the workspace README file."""
    for candidate in README_CANDIDATES:
        path = os.path.join(workspace_dir, candidate)
        if os.path.exists(path) and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return sanitize_readme_for_llm(content)
            except Exception as e:
                console.print(f"[dim]Warning: Failed to read {path}: {e}[/dim]")
    return ""


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

    # 1. Fetch dynamic MCP tool schemas & dispatchers if configured
    mcp_schemas, mcp_dispatchers = await get_mcp_tool_schemas_and_dispatchers()

    # 2. Combine native tools + active MCP tools into unified runtime objects
    active_tools_schema = FILE_TOOLS_SCHEMA + SHELL_TOOLS_SCHEMA + mcp_schemas
    active_tool_dispatchers = {**TOOL_DISPATCHER, **ASYNC_TOOL_DISPATCHER, **mcp_dispatchers}

    # Load dynamic skills, project README documentation, and config manifests
    skills_context = load_project_skills()
    readme_context = read_workspace_readme()
    config_context = load_workspace_config_context()

    system_prompt = (
        "You are an autonomous software engineering agent operating in a CLI workspace.\n\n"
        "--- WORKSPACE CONTEXT & GUIDELINES ---\n"
        f"<skills>\n{skills_context if skills_context else 'No custom skill guidelines provided.'}\n</skills>\n\n"
        f"<readme_documentation>\n{readme_context if readme_context else 'No README file detected in workspace.'}\n</readme_documentation>\n\n"
        f"<workspace_configurations>\n{config_context if config_context else 'No standard package manifests detected.'}\n</workspace_configurations>\n\n"
        "--- CONTEXT PROCESSING & PRIORITIZATION RULES ---\n"
        "1. SKILL GUIDELINES (HIGHEST PRIORITY):\n"
        "   - Directives inside <skills> represent explicit project conventions (architecture, state management, testing patterns).\n"
        "   - Always prioritize <skills> guidelines over generic default assumptions.\n\n"
        "2. README OPERATIONAL METADATA EXTRACTOR:\n"
        "   - Inspect <readme_documentation> exclusively for TECHNICAL OPERATIONAL PARAMETERS.\n"
        "   - EXTRACT ONLY: Exact CLI commands (`pnpm run test`, `uv run pytest`), required package managers (`pnpm`, `uv`, `cargo`), "
        "folder hierarchy layouts, and build prerequisites.\n"
        "   - IGNORE ALL: Marketing text, project overviews, badges, contributor guides, and license blocks.\n"
        "   - FALLBACK: If <readme_documentation> is missing, minimal, or lacks scripts, infer tools directly from <workspace_configurations> instead of guessing.\n\n"
        "VECTOR CONTEXT & FILE DISCOVERY RULES:\n"
        "1. Strictly rely on provided Chroma vector search results injected into your user prompt to locate files and understand project components.\n"
        "2. DO NOT execute terminal shell commands (e.g., `find`, `grep`, `locate`, `ls -R`) to discover or search for files.\n"
        "3. BEFORE modifying or creating any files, verify target component existence within vector context or workspace manifests. "
        "If required files/components are NOT present in vector context, TERMINATE session immediately and inform user.\n"
        "4. DO NOT create duplicate folders or component scaffolding if vector context yields no exact match.\n\n"
        "COMMAND EXECUTION & SAFETY RULES:\n"
        "1. NEVER execute interactive daemons or MCP servers in foreground turns (e.g., `ng mcp`, `ng serve`). "
        "Only run non-blocking CLI commands, or spawn background jobs for builds/tests using `start_background_task` and inspect them using `get_background_task_status`.\n"
        "2. Always inspect package manifests (`package.json`, `pyproject.toml`) before running shell commands.\n"
        "3. Output valid JSON strictly matching tool schema:\n"
        '   {"name": "tool_name", "arguments": {"arg": "value"}}\n'
        "4. If a tool command or build fails, DO NOT repeat identical arguments. Read error output, inspect files, or adjust flags."
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

        # Send the unified active_tools_schema to the LLM client
        response_obj, metrics = await llm_client.chat(messages, tools=active_tools_schema)
        tool_name, raw_args = parse_tool_call(response_obj)

        if tool_name and tool_name in active_tool_dispatchers:
            # Pre-validate arguments through Pydantic self-healing schemas
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

            # Sliding window circuit breaker
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

            # Dispatch using unified active_tool_dispatchers
            tool_result = await handle_tool_call(tool_name, validated_args, active_tool_dispatchers)
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
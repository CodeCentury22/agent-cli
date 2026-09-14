import os
import re
import json
from rich.console import Console
from agent_llm_client import BaseLLMClient
from agent_vector_memory import VectorStoreManager
from agent_vector_memory.sync import sync_workspace_vector_memory
from agent_workspace_tools import (
    WORKSPACE_TOOLS_SCHEMA,
    WORKSPACE_TOOL_DISPATCHER,
    get_git_status_changes,
)
from agent_guardrails import validate_tool_args
from .tool_handler import handle_tool_call
from .agent_workspace import load_project_skills
from .mcp_manager import ensure_and_load_mcp_servers, get_mcp_tool_schemas_and_dispatchers


console = Console()

# REPL-loop robustness budgets (self-healing retry circuits).
MAX_AGENT_TURNS = 15
MAX_VALIDATION_RETRIES = 3
MAX_HALLUCINATION_RETRIES = 3
REPEAT_WINDOW = 6

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
    """Locates, reads, and sanitizes the workspace README file.

    Reads flow through WORKSPACE_TOOL_DISPATCHER (read_file) instead of raw open().
    """
    reader = WORKSPACE_TOOL_DISPATCHER.get("read_file")
    for candidate in README_CANDIDATES:
        path = os.path.join(workspace_dir, candidate)
        if os.path.exists(path) and os.path.isfile(path):
            try:
                if reader is not None:
                    res = reader(file_path=path)
                    if not isinstance(res, dict) or res.get("status") == "ERROR":
                        continue
                    content = (res.get("content", "") or "").strip()
                else:  # pragma: no cover - defensive fallback
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                if content:
                    return sanitize_readme_for_llm(content)
            except Exception as e:
                console.print(f"[dim]Warning: Failed to read {path}: {e}[/dim]")
    return ""


def load_workspace_config_context(workspace_dir: str = ".") -> str:
    """Reads project configuration files to ground the system prompt context.

    Reads flow through WORKSPACE_TOOL_DISPATCHER (read_file) instead of raw open().
    """
    configs = ["package.json", "angular.json", "tsconfig.json"]
    parts = []
    reader = WORKSPACE_TOOL_DISPATCHER.get("read_file")
    for cfg in configs:
        cfg_path = os.path.join(workspace_dir, cfg)
        if os.path.exists(cfg_path):
            try:
                if reader is not None:
                    res = reader(file_path=cfg_path)
                    if not isinstance(res, dict) or res.get("status") == "ERROR":
                        continue
                    content = res.get("content", "") or ""
                else:  # pragma: no cover - defensive fallback
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


def format_memory_results(matches: list | tuple | None, max_chars: int = 9000) -> str:
    """Formats enriched vector-memory hits so the LLM sees line ranges,
    language tags, headings, and relevance scores.

    Expected match keys (upgraded agent-vector-memory): file_path, content,
    relevance_score, distance, start_line, end_line, language, heading.
    """
    if not matches:
        return ""
    blocks: list[str] = []
    total = 0
    for m in matches:
        if not isinstance(m, dict):
            continue
        file_path = m.get("file_path", "") or "unknown"
        start, end = m.get("start_line"), m.get("end_line")
        if isinstance(start, int) and isinstance(end, int):
            location = f"{file_path}#L{start}-{end}"
        elif isinstance(start, int):
            location = f"{file_path}#L{start}"
        else:
            location = file_path
        language = m.get("language") or ""
        heading = m.get("heading") or ""
        score = m.get("relevance_score", 0.0)
        distance = m.get("distance")
        header = f"--- {location} (relevance={score:.2f}"
        if isinstance(distance, (int, float)):
            header += f", distance={distance:.3f}"
        if language:
            header += f", lang={language}"
        if heading:
            header += f", symbol={heading}"
        header += ") ---"
        block = f"{header}\n{m.get('content', '')}".strip()
        if total + len(block) > max_chars and blocks:
            blocks.append("...[memory context truncated for token efficiency]...")
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


async def run_agent_turn(user_input: str, llm_client: BaseLLMClient, vector_store: VectorStoreManager):
    """Executes a complete single-user-request turn with multi-turn tool calling and guardrails.

    Syncs vector memory via agent-workspace-tools git status + incremental
    agent-vector-memory sync, injects enriched (line/language/heading/score)
    memory context, and routes every tool call exclusively through
    WORKSPACE_TOOL_DISPATCHER with guardrail self-healing retries.
    """
    # Incremental memory sync routed through the unified workspace git helper.
    try:
        is_git_repo, files_to_update, files_to_delete = await get_git_status_changes(".")
    except Exception as e:
        console.print(f"[dim]Warning: git status sync skipped: {e}[/dim]")
        is_git_repo, files_to_update, files_to_delete = False, set(), set()
    try:
        sync_workspace_vector_memory(
            vector_store=vector_store,
            workspace_dir=".",
            is_git_repo=is_git_repo,
            files_to_update=files_to_update,
            files_to_delete=files_to_delete,
        )
        if files_to_update or files_to_delete:
            console.print(
                f"[dim]🧠 [Vector Memory]: Synced {len(files_to_update) + len(files_to_delete)} "
                "modified file(s) into Chroma.[/dim]"
            )
    except Exception as e:
        console.print(f"[dim]Warning: vector memory sync failed: {e}[/dim]")
    try:
        # distinct_files=True gives diverse per-file hits for orchestrator context.
        context_matches = vector_store.search_codebase(user_input, top_k=5, distinct_files=True)
    except TypeError:
        context_matches = vector_store.search_codebase(user_input, top_k=5)
    except Exception as e:
        console.print(f"[dim]Warning: vector search failed: {e}[/dim]")
        context_matches = []
    context_str = format_memory_results(context_matches) or "No relevant codebase context found."

    # 1. Fetch dynamic MCP tool schemas & dispatchers if configured
    mcp_schemas, mcp_dispatchers = await get_mcp_tool_schemas_and_dispatchers()

    # Define the explicit DONE tool
    done_tool_schema = [{
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call this tool with empty arguments when the task is fully complete and verified.",
            "parameters": {"type": "object", "properties": {}}
        }
    }]

    # 2. Combine native tools + active MCP tools + done tool into unified runtime objects
    active_tools_schema = WORKSPACE_TOOLS_SCHEMA + mcp_schemas + done_tool_schema
    active_tool_dispatchers = {**WORKSPACE_TOOL_DISPATCHER, **mcp_dispatchers}

    # Load dynamic skills, project README documentation, and config manifests
    skills_context = load_project_skills()
    readme_context = read_workspace_readme()
    config_context = load_workspace_config_context()

    system_prompt = (
        "You are an autonomous software engineering agent operating in a CLI workspace.\n\n"
        "--- CRITICAL TOOL EXECUTION PROTOCOL ---\n"
        "1. When performing tasks that require modifying files, running tests, or building, YOU MUST INVOKE TOOLS.\n"
        "2. DO NOT write out file contents, CSS code blocks, or explanations in your final text response when a tool is needed.\n"
        "3. To invoke a tool, your entire output must culminate in or consist strictly of a JSON object matching this schema:\n"
        '   {"name": "tool_name", "arguments": {"arg": "value"}}\n'
        "4. When the overarching task is completely finished and verified, you MUST output:\n"
        '   {"name": "done", "arguments": {}}\n'
        "5. Never output markdown code blocks containing code changes or file contents if a file-writing tool is available. Execute the tool instead.\n\n"
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
        "Only run non-blocking CLI commands, or spawn background jobs for builds/tests using `start_background_task` and inspect them using `get_background_task_status`. "
        "Once `get_background_task_status` reports a `SUCCESS` status, STOP polling status, present the successful outcome to the user, and conclude the turn.\n"
        "2. Always inspect package manifests (`package.json`, `pyproject.toml`) before running shell commands.\n"
        "3. Output valid JSON strictly matching tool schema:\n"
        '   {"name": "tool_name", "arguments": {"arg": "value"}}\n'
        "4. If a tool command or build fails, DO NOT repeat identical arguments. Read error output, inspect files, or adjust flags."
    )


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nTask: {user_input}"}
    ]

    recent_tool_signatures: list[tuple[str, str]] = []
    validation_retries = 0
    hallucination_retries = 0

    for _turn in range(MAX_AGENT_TURNS):
        # Prune old context messages to avoid token bloat during deep turns
        if len(messages) > 12:
            messages = [messages[0], messages[1]] + messages[-8:]

        # Send the unified active_tools_schema to the LLM client
        try:
            response_obj, metrics = await llm_client.chat(messages, tools=active_tools_schema)
        except Exception as e:
            console.print(f"\n❌ [LLM Error]: {e}. Ending turn.")
            break
        tool_name, raw_args = parse_tool_call(response_obj)

        if tool_name == "done":
            console.print("\n🎉 [bold green]Agent declared task complete![/bold green]")
            break

        # --- Robust hallucination intercept: unknown tool names get a repair nudge,
        # --- not a crash, with a bounded self-healing retry budget.
        if tool_name and tool_name not in active_tool_dispatchers:
            hallucination_retries += 1
            known = ", ".join(sorted(active_tool_dispatchers))
            console.print(
                f"\n⚠️ [Unknown Tool '{tool_name}']: model hallucinated an unregistered tool "
                f"({hallucination_retries}/{MAX_HALLUCINATION_RETRIES}). Requesting repair..."
            )
            messages.append({"role": "assistant", "content": str(response_obj)})
            if hallucination_retries >= MAX_HALLUCINATION_RETRIES:
                console.print(
                    "\n🛑 [Circuit Breaker]: too many unknown-tool hallucinations. Halting turn."
                )
                break
            messages.append({
                "role": "user",
                "content": (
                    f"System Error: Tool '{tool_name}' is not registered. "
                    f"Available tools: {known}. "
                    "Reply with a corrected JSON tool call using only a registered tool name, "
                    "or call 'done' if finished."
                ),
            })
            continue
        hallucination_retries = 0

        if tool_name and tool_name in active_tool_dispatchers:
            # Pre-validate arguments through Pydantic self-healing schemas
            is_valid, validated_args, err_msg = validate_tool_args(tool_name, raw_args)

            if not is_valid:
                validation_retries += 1
                console.print(
                    f"\n⚠️ [Schema Validation Error]: {err_msg} "
                    f"({validation_retries}/{MAX_VALIDATION_RETRIES}). Triggering prompt repair..."
                )
                messages.append({"role": "assistant", "content": str(response_obj)})
                if validation_retries >= MAX_VALIDATION_RETRIES:
                    console.print(
                        "\n🛑 [Circuit Breaker]: schema validation failed repeatedly. Halting turn."
                    )
                    break
                messages.append({
                    "role": "user",
                    "content": f"System Error: Tool call '{tool_name}' arguments were invalid: {err_msg}. Please fix the arguments according to the schema and retry."
                })
                continue
            validation_retries = 0

            tool_signature = (tool_name, json.dumps(validated_args, sort_keys=True, default=str))

            # Sliding window circuit breaker (exempt background task polling from strict loop interruption)
            if tool_name not in ["start_background_task", "get_background_task_status"]:
                if recent_tool_signatures.count(tool_signature) >= 2:
                    console.print(f"\n🛑 [Circuit Breaker]: Detected repeating tool call loop for '{tool_name}'. Halting turn.")
                    messages.append({
                        "role": "user",
                        "content": f"System Warning: Stop repeating command '{tool_name}'. The file is already written or the command failed. Call 'done' or proceed to the next task step."
                    })
                    break

                recent_tool_signatures.append(tool_signature)
                if len(recent_tool_signatures) > REPEAT_WINDOW:
                    recent_tool_signatures.pop(0)

            console.print(f"\n🛠️  [bold yellow]Agent Invoking Tool:[/bold yellow] [cyan]{tool_name}[/cyan]")

            # Dispatch exclusively through the unified WORKSPACE_TOOL_DISPATCHER (+ MCP).
            try:
                tool_result = await handle_tool_call(tool_name, validated_args, active_tool_dispatchers)
            except Exception as e:
                tool_result = {"status": "ERROR", "error": f"Dispatcher exception for '{tool_name}': {e}"}
            console.print(f"📋 [bold green]Tool Execution Result:[/bold green]\n{tool_result}")

            messages.append({"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": validated_args})})
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' Output:\n{tool_result}\n\nContinue with task or call next tool. If finished, call 'done'."
            })

            console.print(f"[dim]Metrics: {metrics}[/dim]")
            continue

        console.print(f"\n🤖 [bold cyan]Agent Response:[/bold cyan]\n{response_obj}")
        console.print(f"\n[dim]Metrics: {metrics}[/dim]")
        break
    else:
        console.print(
            f"\n🛑 [Turn Budget]: reached MAX_AGENT_TURNS ({MAX_AGENT_TURNS}). Halting turn."
        )
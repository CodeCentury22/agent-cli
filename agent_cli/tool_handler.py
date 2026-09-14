import asyncio
from rich.console import Console

from agent_guardrails import validate_tool_args
from agent_workspace_tools import WORKSPACE_TOOL_DISPATCHER
from .diff_viewer import render_file_diff

console = Console()

async def handle_tool_call(tool_name: str, raw_args: dict, dispatchers: dict) -> dict:
    """Sanitizes arguments via guardrails, displays diffs for file edits, and dispatches tools.

    All execution routes exclusively through WORKSPACE_TOOL_DISPATCHER
    (merged with MCP dispatchers at runtime); this module never touches the
    filesystem or subprocesses directly. Diff previews are read back through
    the same dispatcher for consistency.
    """
    is_valid, sanitized_args, error_msg = validate_tool_args(tool_name, raw_args)

    if not is_valid:
        console.print(f"❌ [Guardrail Reject]: {error_msg}")
        return {"status": "ERROR", "error": error_msg}

    if tool_name == "write_file":
        target_path = sanitized_args.get("file_path")
        new_code = sanitized_args.get("code_body", "")
        old_code = ""

        if target_path:
            try:
                reader = dispatchers.get("read_file") or WORKSPACE_TOOL_DISPATCHER.get("read_file")
                preview = reader(file_path=target_path)
                if asyncio.iscoroutine(preview):
                    preview = await preview
                if isinstance(preview, dict) and preview.get("status") != "ERROR":
                    old_code = preview.get("content", "") or ""
            except Exception:
                old_code = ""

            render_file_diff(target_path, old_code, new_code)

    dispatcher = dispatchers.get(tool_name)
    if not dispatcher:
        available = ", ".join(sorted(dispatchers)) if dispatchers else "none"
        return {
            "status": "ERROR",
            "error": f"Tool '{tool_name}' not registered. Available tools: {available}",
        }

    try:
        result = dispatcher(**sanitized_args)
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as e:
        return {"status": "ERROR", "error": f"Execution failed for '{tool_name}': {e}"}

    return result
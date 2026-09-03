import os
import asyncio
from rich.console import Console

from agent_guardrails import validate_tool_args
from .diff_viewer import render_file_diff

console = Console()

async def handle_tool_call(tool_name: str, raw_args: dict, dispatchers: dict) -> dict:
    """Sanitizes arguments via guardrails, displays diffs for file edits, and dispatches tools."""
    is_valid, sanitized_args, error_msg = validate_tool_args(tool_name, raw_args)
    
    if not is_valid:
        console.print(f"❌ [Guardrail Reject]: {error_msg}")
        return {"status": "ERROR", "error": error_msg}

    if tool_name == "write_file":
        target_path = sanitized_args.get("file_path")
        new_code = sanitized_args.get("code_body", "")
        old_code = ""

        if target_path and os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    old_code = f.read()
            except Exception:
                old_code = ""

        if target_path:
            render_file_diff(target_path, old_code, new_code)

    dispatcher = dispatchers.get(tool_name)
    if not dispatcher:
        return {"status": "ERROR", "error": f"Tool '{tool_name}' not registered."}

    result = dispatcher(**sanitized_args)
    if asyncio.iscoroutine(result):
        result = await result

    return result
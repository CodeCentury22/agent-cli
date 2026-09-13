# Agent Capabilities & Engineering Guidelines

## 1. File Modification Rules (CRITICAL)
- **NEVER** use `write_file` to modify an existing file. `write_file` is strictly reserved for generating brand-new files.
- When updating existing code, styles, or templates, you **MUST** use `replace_in_file` to perform precise string replacements.
- Always inspect the file using `read_file` first to ensure exact matching of indentation, whitespace, and code blocks.

## 2. Shell Execution & Background Tasks
- **DO NOT** launch interactive dev servers or blocking daemons (e.g., `ng serve`, `vite`, `flutter run`) via standard `run_shell_command`.
- For builds, test suites, or long-running checks, prefer `start_background_task` and monitor progress using `get_background_task_status` to maintain agent loop momentum.

## 3. Vector Memory & ChromaDB Integration
- The workspace is indexed using a local **ChromaDB** vector memory module containing embeddings of all project files, components, and architectural decisions.
- If you are unsure where a component, service, or CSS class is located across the monorepo, rely on codebase search tools or memory context rather than guessing paths.
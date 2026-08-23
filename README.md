# 🤖 Agent CLI (`agent-cli`)

A modular, zero-argument interactive REPL terminal for local AI software development. Built for developers who want a local coding assistant backed by multi-provider LLMs, persistent authentication, workspace vector memory, and automated code-editing tools.

## 🚀 Quick Install

Install `agent-cli` along with all prerequisites (`python`, `git`, and `uv`) using a single terminal command.

### Linux / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/CodeCentury22/agent-cli/main/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/CodeCentury22/agent-cli/main/install.ps1 | iex
```

> **Note:** The installer automatically configures Astral `uv`, installs `agent-cli` as a global CLI tool, and updates your environment `PATH`.

## 🌟 Key Features

* **Zero-Argument REPL:** Run `agent-cli` anywhere in your terminal to start an interactive session.
* **Multi-Provider LLM Gateway:** Switch seamlessly between local models (Ollama) and cloud providers (Google Gemini, Anthropic Claude, and OpenRouter).
* **Workspace Vector Memory:** Automatically indexes your local codebase into ChromaDB using semantic code chunking for deep context retrieval.
* **Persistent Credential Manager:** Securely stores API keys and handles browser OAuth callbacks locally in `~/.config/agent-cli/credentials.json`.
* **Cross-Platform POSIX Normalization:** Fully OS-aware and tested across Linux, macOS, and Windows.

## 🛠️ Usage

### Launch Interactive Session

Navigate to any codebase directory and launch the CLI:

```bash
agent-cli
```

### Session Wizard & Provider Selection

Upon launch, `agent-cli` guides you through provider and model selection:

```text
==================================================
   🤖 Agent CLI - Interactive Workspace Session
==================================================

Select LLM Provider:
    1) Local (Ollama)
    2) Google Gemini
    3) Anthropic Claude
    4) OpenRouter / OpenCode Gateway

Choose provider [1]: 3

Select Model for Anthropic Claude:
    1) claude-3-5-sonnet-20241022
    2) claude-3-5-haiku-20241022

Choose model [1]: 1

🔑 Using stored credentials for claude.

Session Active!
Directory: /home/user/projects/my-app
Provider:  claude
Model:     claude-3-5-sonnet-20241022

Type 'exit' or 'quit' to end the session.

agent> Refactor the auth handling in main.py to handle token expiration safely.
```

## 🏗️ Ecosystem Micro-Libraries

`agent-cli` is engineered on top of a decoupled micro-library architecture:

| Component               | Repository            | Description                                                                        |
| ----------------------- | --------------------- | ---------------------------------------------------------------------------------- |
| **Unified LLM Gateway** | `agent-llm-client`    | Standardized driver for Ollama, Gemini, Claude, and OpenRouter with token metrics. |
| **Semantic Memory**     | `agent-vector-memory` | Codebase chunking, AST extraction, and ChromaDB vector search.                     |
| **File Manipulation**   | `agent-file-tools`    | POSIX-normalized file operations (`read`, `write`, `delete`, `list`).              |
| **Output Guardrails**   | `agent-guardrails`    | Structured JSON schema validation and auto-retry logic for LLM payloads.           |
| **Async Task Runner**   | `agent-async-runner`  | Concurrent task execution loops.                                                   |
| **Core Utilities**      | `agent-core-utils`    | Latency tracking and JSONL audit logging decorators.                               |

## 💻 Local Development & Testing

If you want to contribute or build from source using `uv`:

```bash
# 1. Clone repo
git clone https://github.com/CodeCentury22/agent-cli.git
cd agent-cli

# 2. Sync virtual environment and lockfile
uv sync

# 3. Run unit test suite
uv run pytest

# 4. Launch local build
uv run agent-cli
```

## 📄 License

MIT License © CodeCentury22

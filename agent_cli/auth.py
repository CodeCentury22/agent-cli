import json
import os
import webbrowser
import http.server
import socketserver
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".config" / "agent-cli"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"

def get_stored_credentials(provider: str) -> Optional[str]:
    """Retrieves stored API key or OAuth token for a provider if it exists."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        return data.get(provider)
    except (json.JSONDecodeError, OSError):
        return None

def save_credentials(provider: str, secret: str) -> None:
    """Saves API key or OAuth token locally in ~/.config/agent-cli/credentials.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if CREDENTIALS_FILE.exists():
        try:
            data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

    data[provider] = secret
    CREDENTIALS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def start_browser_oauth_flow(auth_url: str, port: int = 8080) -> Optional[str]:
    """Spins up a local HTTP server, launches browser auth, and captures the callback code."""
    auth_code: Optional[str] = None

    class OAuthCallbackHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            if "?code=" in self.path:
                auth_code = self.path.split("?code=")[1].split("&")[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")  # Fixed send_head typo
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Authentication Successful!</h1>"
                    b"<p>You may close this browser window and return to agent-cli.</p>"
                    b"</body></html>"
                )
            else:
                self.send_response(400)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    print(f"\n🌐 Opening browser for authentication...\nURL: {auth_url}\n")
    webbrowser.open(auth_url)

    with socketserver.TCPServer(("127.0.0.1", port), OAuthCallbackHandler) as httpd:
        httpd.handle_request()

    return auth_code
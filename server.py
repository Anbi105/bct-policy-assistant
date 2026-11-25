"""
STEP 2 - Start the web UI.

    python server.py          ->  opens http://127.0.0.1:8000 in your browser

Uses only Python's built-in web server (no Flask / Streamlit needed).
Answers are streamed to the page word by word, so it feels instant.
"""
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
import rag

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

INDEX_HTML = (config.ROOT / "ui" / "index.html").read_bytes()
retriever = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):      # keep the terminal clean
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/info":
            info = {
                "provider": config.PROVIDER,
                "model": config.LLM_MODEL,
                "search": retriever.mode,
                "chunks": len(retriever.chunks),
                "has_key": bool(config.LLM_API_KEY) or config.PROVIDER == "ollama",
            }
            self._send(200, json.dumps(info).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/ask":
            return self._send(404, b'{"error":"not found"}')
        try:
            length = int(self.headers.get("Content-Length", 0))
            question = json.loads(self.rfile.read(length) or b"{}").get("question", "").strip()
        except Exception:
            return self._send(400, b'{"error":"bad request"}')
        if not question:
            return self._send(400, b'{"error":"empty question"}')

        # Stream newline-delimited JSON events to the browser.
        # (HTTP/1.0: the stream ends when we close the connection.)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for event in rag.ask(question):
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global retriever
    print("Loading knowledge base...")
    retriever = rag.get_retriever()
    retriever.warm_up()
    print(f"  {len(retriever.chunks)} chunks | search: {retriever.mode}")
    print(f"  LLM: {config.PROVIDER} / {config.LLM_MODEL}")
    if config.PROVIDER != "ollama" and not config.LLM_API_KEY:
        print(f"  [!] No API key. Add {config.KEY_ENV_NAME}=... to your .env file, then restart.")

    server = ThreadingHTTPServer(("127.0.0.1", config.PORT), Handler)
    url = f"http://127.0.0.1:{config.PORT}"
    print(f"\nReady -> {url}   (press Ctrl+C to stop)\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

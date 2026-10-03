"""Stand-in for llama-server used by test_llamacpp_server.py.

Accepts the same flags the adapter passes, serves /health, /props, /v1/chat/completions and a streaming
/completion with a fixed time to first token and a fixed time per output token, so the adapter's
TTFT/TPOT measurement can be checked against known values. ``-m`` pointing at a missing file reproduces
llama-server's load-error exit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TTFT = 0.10
TPOT = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("-t")
    ap.add_argument("-c", type=int)
    ap.add_argument("-np", type=int, default=1)
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--no-webui", action="store_true")
    a, _ = ap.parse_known_args()
    if not Path(a.m).is_file():
        print(f"gguf_init_from_file: failed to open GGUF file '{a.m}'", flush=True)
        print(f"srv    load_model: failed to load model, '{a.m}'", flush=True)
        sys.exit(1)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def _json(self, obj: object) -> None:
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json({"status": "ok"})
            elif self.path == "/props":
                self._json({"total_slots": a.np, "build_info": "fake", "default_generation_settings": {"n_ctx": a.c // a.np}})
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/chat/completions":
                self._json({"choices": [{"message": {"content": "Paris"}}], "usage": {"completion_tokens": 1}})
                return
            if self.path != "/completion":
                self.send_error(404)
                return
            n = int(body["n_predict"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            time.sleep(TTFT)
            for i in range(n):
                if i:
                    time.sleep(TPOT)
                self.wfile.write(b"data: " + json.dumps({"content": "x", "tokens": [7], "stop": False}).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: " + json.dumps({"content": "", "tokens": [], "stop": True}).encode() + b"\n\n")
            self.wfile.flush()
            self.close_connection = True

    time.sleep(0.3)  # model "loading": the adapter must poll /health until it answers
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()

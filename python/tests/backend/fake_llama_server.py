"""Mock serving backend for the deployment / load-test tests (T-0703, T-0705, T-0803).

Two launch styles, both on 127.0.0.1:

* llama-server style: ``-m MODEL --port P -c N -np K ...`` (serves ``/health`` as JSON, ``/props``);
* vLLM style: ``serve MODEL --port P --max-model-len N --max-num-seqs K ...`` (``/health`` is an empty 200,
  ``/v1/models`` lists the model).

Both serve ``/v1/chat/completions``, a streaming ``/v1/completions`` (SSE with a usage chunk and ``[DONE]``)
and, llama-style, a streaming ``/completion``. Token i of a stream is sent at ``t0 + ttft + i * tpot``
(absolute deadlines, TCP_NODELAY), so a load tester can be checked against known latencies.

Failure switches, picked from the model path: a missing file exits like the real server would; a name containing
``oom`` / ``badquant`` prints the real error text and exits 1. ``FAKE_OOM_ABOVE_SEQS=N`` makes a vLLM-style launch
with ``--max-num-seqs`` above N die with an out-of-memory error (for boundary-search tests).
Latencies: ``--ttft`` / ``--tpot`` seconds (defaults 0.10 / 0.02).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _sleep_until(deadline: float) -> None:
    """Wait until ``deadline`` (perf_counter). time.sleep overshoots by 1-2 ms on macOS, which would show up as
    latency error in the load tester, so only the coarse part is slept and the last 3 ms are spun."""
    while (left := deadline - time.perf_counter()) > 0:
        if left > 0.004:
            time.sleep(left - 0.003)


def main() -> None:
    argv = sys.argv[1:]
    vllm_style = bool(argv) and argv[0] == "serve"
    if vllm_style:
        argv = argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("model_pos", nargs="?")
    ap.add_argument("-m")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("-c", type=int, default=4096)
    ap.add_argument("-np", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=0)
    ap.add_argument("--served-model-name")
    ap.add_argument("--ttft", type=float, default=0.10)
    ap.add_argument("--tpot", type=float, default=0.02)
    a, _ = ap.parse_known_args(argv)
    model = a.model_pos if vllm_style else a.m
    max_len = a.max_model_len or a.c // max(1, a.np)
    name = a.served_model_name or model

    if vllm_style:
        if not Path(model).exists():
            print(f"OSError: Invalid repository ID or local directory specified: '{model}'", flush=True)
            sys.exit(1)
        if "oom" in Path(model).name:
            print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", flush=True)
            sys.exit(1)
        if "badquant" in Path(model).name:
            print("pydantic_core.ValidationError: Value error, Unsupported quantization config: bits=4, sym=False", flush=True)
            sys.exit(1)
        cap = int(os.environ.get("FAKE_OOM_ABOVE_SEQS", "0"))
        if cap and a.max_num_seqs > cap:
            print(f"torch.OutOfMemoryError: CUDA out of memory (max_num_seqs={a.max_num_seqs} > {cap})", flush=True)
            sys.exit(1)
        len_cap = int(os.environ.get("FAKE_MAX_LEN_CAP", "0"))
        if len_cap and a.max_model_len > len_cap:
            print(f"ValueError: The model's max seq len ({a.max_model_len}) is larger than the maximum number of tokens "
                  f"that can be stored in KV cache ({len_cap}).", flush=True)
            sys.exit(1)
        print("Model loading took 0.43 GiB memory and 2.1 seconds", flush=True)
        print("GPU KV cache size: 12,345 tokens", flush=True)
        print(f"Maximum concurrency for {max_len:,} tokens per request: {12345 / max_len:.2f}x", flush=True)
    elif not Path(model).is_file():
        print(f"gguf_init_from_file: failed to open GGUF file '{model}'", flush=True)
        print(f"srv    load_model: failed to load model, '{model}'", flush=True)
        sys.exit(1)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def _send(self, code: int, payload: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, obj: object, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self) -> None:
            if self.path == "/health":
                if vllm_style:
                    self._send(200, b"")
                else:
                    self._json({"status": "ok"})
            elif self.path == "/props" and not vllm_style:
                self._json({"total_slots": a.np, "build_info": "fake", "default_generation_settings": {"n_ctx": a.c // a.np}})
            elif self.path == "/v1/models":
                self._json({"data": [{"id": name, "max_model_len": max_len}]})
            else:
                self.send_error(404)

        def _stream(self, n: int, chunk: bytes, tail: list[bytes]) -> None:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t0 = time.perf_counter()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(n):
                _sleep_until(t0 + a.ttft + i * a.tpot)
                self.wfile.write(chunk)
                self.wfile.flush()
            for t in tail:
                self.wfile.write(t)
            self.wfile.flush()
            self.close_connection = True

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/chat/completions":
                self._json({"choices": [{"message": {"content": "Paris"}}], "usage": {"completion_tokens": 1}})
            elif self.path == "/v1/completions" and not body.get("stream"):
                choice: dict[str, object] = {"text": "x"}
                if vllm_style and body.get("prompt_logprobs") is not None:
                    # scoring request: every prompt token after the first gets logprob FAKE_LOGPROB (llama-server has
                    # no prompt_logprobs, so the llama style answers without the field)
                    lp = float(os.environ.get("FAKE_LOGPROB", "-2.0"))
                    choice["prompt_logprobs"] = [None] + [{str(t): {"logprob": lp, "rank": 1}} for t in body["prompt"][1:]]
                self._json({"choices": [choice]})
            elif self.path == "/v1/completions":
                n, plen = int(body.get("max_tokens", 16)), len(body.get("prompt") or [])
                if plen + n > max_len:
                    self._json({"error": {"message": f"This model's maximum context length is {max_len} tokens"}}, 400)
                    return
                chunk = b"data: " + json.dumps({"choices": [{"text": "x", "finish_reason": None}]}).encode() + b"\n\n"
                usage = {"choices": [], "usage": {"prompt_tokens": plen, "completion_tokens": n, "total_tokens": plen + n}}
                self._stream(n, chunk, [b"data: " + json.dumps(usage).encode() + b"\n\n", b"data: [DONE]\n\n"])
            elif self.path == "/completion" and not vllm_style:
                n = int(body["n_predict"])
                chunk = b"data: " + json.dumps({"content": "x", "tokens": [7], "stop": False}).encode() + b"\n\n"
                self._stream(n, chunk, [b"data: " + json.dumps({"content": "", "tokens": [], "stop": True}).encode() + b"\n\n"])
            else:
                self.send_error(404)

    time.sleep(0.3)  # model "loading": the adapter must poll /health until it answers
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()

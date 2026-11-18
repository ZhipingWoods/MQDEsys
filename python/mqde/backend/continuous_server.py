"""Native HTTP serving for continuous-output models (VLA actions, WM/EWM video).

vLLM / SGLang / llama.cpp expose token generation. VLA and world models output action vectors or
video frames, so they need a different protocol. This module runs a loopback HTTP server inside the
job process, on top of the already-loaded (and, for a deployment stage, already fake-quantized)
adapter:

    GET  /health          readiness
    GET  /v1/models       served model id / domain
    POST /v1/actions      VLA: images + instruction + state -> action chunk
    POST /v1/video        WM/EWM: prompt (+ optional actions) -> frames

The server is intentionally dependency-free (stdlib http.server), binds 127.0.0.1 only, and is
stopped before the next recipe restores the FP weights.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from mqde.backend.base import HOST, free_port


def _png_b64(arr: np.ndarray) -> str:
    from PIL import Image

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8) if a.max(initial=0.0) <= 1.0 else a.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_png(s: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


class _Handler(BaseHTTPRequestHandler):
    server_version = "MQDE-native/1.0"

    @property
    def owner(self) -> "ContinuousServer":
        return self.server.owner  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": self.owner.model_name, "domain": self.owner.domain}]})
        if self.path == "/v1/protocol":
            return self._send(200, {"domain": self.owner.domain, "action_dim": self.owner.action_dim,
                                    "frames": self.owner.frames, "protocol": self.owner.protocol})
        return self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad request: {e}"})
        try:
            if self.path == "/v1/actions" and self.owner.domain == "vla":
                return self._send(200, self.owner.actions(body))
            if self.path == "/v1/video" and self.owner.domain in ("world", "ewm"):
                return self._send(200, self.owner.video(body))
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


class ContinuousServer:
    """Loopback server around one VLA or world-model adapter."""

    def __init__(self, adapter: Any, domain: str, *, model_name: str = "mqde-model", port: int | None = None,
                 action_dim: int = 0, frames: int = 4) -> None:
        self.adapter = adapter
        self.domain = domain
        self.model_name = model_name
        self.port = port or free_port()
        self.action_dim = action_dim
        self.frames = frames
        self.protocol = "vla/actions-v1" if domain == "vla" else "world/video-v1"
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.startup_s: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def start(self) -> "ContinuousServer":
        t0 = time.perf_counter()
        self._httpd = ThreadingHTTPServer((HOST, self.port), _Handler)
        self._httpd.owner = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="mqde-native-server", daemon=True)
        self._thread.start()
        self.startup_s = time.perf_counter() - t0
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def __enter__(self) -> "ContinuousServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ---------------------------------------------------------------- model calls
    def actions(self, body: dict[str, Any]) -> dict[str, Any]:
        from mqde.models.minicpm_vla import VLASample

        images = [_decode_png(x) for x in (body.get("images") or [])]
        if not images:
            raise ValueError("images must contain at least one base64 PNG")
        state = np.asarray(body.get("state") or [], dtype=np.float32)
        sample = VLASample(images=images, text=str(body.get("instruction") or body.get("text") or "complete the task"),
                           state=state, embodiment_id=int(body.get("embodiment_id", 0)))
        seed = int(body.get("seed", 0))
        t0 = time.perf_counter()
        with self._lock:
            out = self.adapter.predict(sample, seed=seed)
        actions = out["actions"].detach().cpu()
        return {"model": self.model_name, "actions": actions.tolist(), "horizon": int(actions.shape[0]),
                "action_dim": int(actions.shape[1]), "latency_ms": (time.perf_counter() - t0) * 1e3}

    def video(self, body: dict[str, Any]) -> dict[str, Any]:
        from mqde.data.world import WorldSample

        n = int(body.get("frames") or self.frames)
        raw_frames = body.get("input_frames") or body.get("frames_b64")
        if raw_frames:
            frames = np.stack([np.asarray(_decode_png(x), dtype=np.float32) / 255.0 for x in raw_frames])
        else:
            frames = np.zeros((n, 32, 32, 3), dtype=np.float32)
        actions = np.asarray(body["actions"], dtype=np.float32) if body.get("actions") is not None else None
        sample = WorldSample(frames=frames, prompt=str(body.get("prompt") or "a short video"),
                             actions=actions, meta={"source": "native-server"})
        seed, steps = int(body.get("seed", 0)), int(body.get("steps") or 0) or None
        t0 = time.perf_counter()
        with self._lock:
            out = self.adapter.generate(sample, seed=seed, steps=steps)
        out = out.detach().cpu().float()
        latency = (time.perf_counter() - t0) * 1e3
        result: dict[str, Any] = {"model": self.model_name, "shape": list(out.shape), "latency_ms": latency}
        if body.get("return_frames", True):
            arr = (out.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
            result["frames"] = [_png_b64(x) for x in arr]
        return result


def _post(port: int, path: str, body: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
    import http.client

    c = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"POST {path} -> HTTP {r.status}: {data[:300]!r}")
        return json.loads(data) if data else {}
    finally:
        c.close()


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def smoke_and_load(adapter: Any, domain: str, cfg: dict[str, Any] | None = None, *, model_name: str = "mqde-model",
                  action_conditioned: bool = False, frames: int = 4) -> dict[str, Any]:
    """Start the native server, run one smoke request and a closed-loop load test, then stop it."""
    cfg = dict(cfg or {})
    conc = max(1, int(cfg.get("concurrency", 2)))
    requests = max(1, int(cfg.get("requests", 4)))
    seed = int(cfg.get("seed", 0))
    srv = ContinuousServer(adapter, domain, model_name=model_name, action_dim=int(cfg.get("action_dim", 0)), frames=frames)
    try:
        srv.start()
        if domain == "vla":
            from mqde.data.vla import synthetic

            s = synthetic(1, seed=seed)[0]
            payload = {"images": [_png_b64(x) for x in s.images], "instruction": s.text,
                       "state": s.state.tolist(), "seed": seed}
            path = "/v1/actions"
            smoke = _post(srv.port, path, payload)
        else:
            from mqde.data.world import synthetic

            sample = synthetic(1, seed=seed, frames=frames, action_conditioned=action_conditioned)[0]
            payload = {"prompt": sample.prompt, "frames": frames, "seed": seed,
                       "actions": sample.actions.tolist() if sample.actions is not None else None,
                       "return_frames": True}
            path = "/v1/video"
            smoke = _post(srv.port, path, payload)
        load_payload = {**payload, "return_frames": False}

        def one(i: int) -> tuple[bool, float, str]:
            t0 = time.perf_counter()
            try:
                _post(srv.port, path, {**load_payload, "seed": seed + i + 1})
                return True, time.perf_counter() - t0, ""
            except Exception as e:
                return False, time.perf_counter() - t0, f"{type(e).__name__}: {e}"

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            rows = list(pool.map(one, range(requests)))
        wall = time.perf_counter() - t0
        ok = [r for r in rows if r[0]]
        lat = [r[1] * 1e3 for r in ok]
        result = {
            "status": "succeeded" if len(ok) == requests else "degraded",
            "protocol": srv.protocol,
            "endpoint": srv.base_url,
            "startup_s": srv.startup_s,
            "smoke": {"ok": True, "latency_ms": smoke.get("latency_ms"), "shape": smoke.get("shape")},
            "load": {
                "ok": len(ok), "requests": requests, "failed": requests - len(ok), "concurrency": conc,
                "wall_s": wall, "requests_per_s": len(ok) / wall if wall else 0.0,
                "latency_ms": {"p50": _pct(lat, 50), "p95": _pct(lat, 95), "mean": (sum(lat) / len(lat)) if lat else None},
                "errors": [r[2] for r in rows if not r[0]][:5],
            },
        }
        return result
    finally:
        srv.stop()

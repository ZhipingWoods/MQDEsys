#!/usr/bin/env python3
"""Stand-in for scripts/setup_backend.py for UI tests that merely visit the hardware page: `mqde serve` runs it through
MQDE_SETUP_BACKEND_BIN, so no environment is inspected or installed. Always reports every backend as installed and current."""

import json
import sys

if sys.argv[1:2] == ["status"]:
    rows = [{"id": b, "installed": True, "version": "1.0.0", "path": f"/envs/{b}", "source": "managed", "managed": True, "envs_dir": "/envs"}
            for b in ("vllm", "sglang", "llamacpp")]
    if "--check" in sys.argv:
        for r in rows:
            r.update(latest="1.0.0", update_available=False)
    print(json.dumps(rows))
else:
    print("the fake setup script installs nothing", file=sys.stderr)
    sys.exit(1)

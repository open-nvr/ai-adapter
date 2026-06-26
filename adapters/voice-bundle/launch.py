#!/usr/bin/env python3
# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Voice-bundle launcher: run the Whisper, Piper, and BLIP adapter apps in ONE
container, each on its existing port, with the SAME per-adapter `/infer`
contract the camera-agent already speaks.

This is the low-risk "combined voice adapter" (issue #82): it collapses three
containers/images into one (fewer things to pull, start, and health-check). It
does NOT use the monolithic ``app.main`` (port 9100, task-routed) — that's a
different contract. The big RAM lever is the editions + model choices, not this
merge; the win here is operational simplicity and one image.

Each app runs as its own uvicorn subprocess so a crash in one is contained; if
any process exits, the whole bundle shuts down so the orchestrator restarts it
cleanly (no silent half-up container).

Override ports / disable an app via env: PIPER_PORT, WHISPER_PORT, BLIP_PORT
(set a port to "0" or "off" to skip that app).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

# (name, ASGI app path, default port, env var that overrides the port)
_APPS = [
    ("piper", "adapters.piper.main:app", "9001", "PIPER_PORT"),
    ("whisper", "adapters.whisper.main:app", "9003", "WHISPER_PORT"),
    ("blip", "adapters.blip.main:app", "9006", "BLIP_PORT"),
]

_procs: list[tuple[str, subprocess.Popen]] = []
_shutting_down = False


def _spawn() -> None:
    for name, app_path, default_port, env_key in _APPS:
        port = os.getenv(env_key, default_port).strip().lower()
        if port in ("", "0", "off", "false", "no"):
            print(f"[voice-bundle] {name}: disabled (port={port!r})", flush=True)
            continue
        cmd = [sys.executable, "-m", "uvicorn", app_path,
               "--host", "0.0.0.0", "--port", port]
        print(f"[voice-bundle] starting {name} on :{port}", flush=True)
        _procs.append((name, subprocess.Popen(cmd)))


def _shutdown(*_a) -> None:
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    for name, p in _procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 10
    for name, p in _procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.time()))
        except Exception:
            p.kill()
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    _spawn()
    if not _procs:
        print("[voice-bundle] no apps enabled; exiting", file=sys.stderr, flush=True)
        sys.exit(1)
    # Supervise: if any child dies, take the whole bundle down so the
    # orchestrator restarts a clean container rather than leaving it half-up.
    while True:
        for name, p in _procs:
            rc = p.poll()
            if rc is not None:
                print(f"[voice-bundle] {name} exited rc={rc}; shutting down bundle",
                      file=sys.stderr, flush=True)
                _shutdown()
        time.sleep(2)


if __name__ == "__main__":
    main()

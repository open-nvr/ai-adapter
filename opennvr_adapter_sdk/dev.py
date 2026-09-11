# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
``opennvr-adapter dev`` — exercise an adapter without a stack.

Between writing a model wrapper and knowing whether it works there used
to be Docker, KAI-C, core and a camera. This runs the adapter
in-process and drives its own contract endpoints, so the first feedback
loop is a second long::

    $ opennvr-adapter dev
    opennvr-adapter dev — fall-detection 1.0.0
      loading the model…                                  ok (0.3s)
      GET  /health                 ok      model_loaded=true
      GET  /capabilities           ok      tasks: object_detection
      GET  /hardware/evaluation    ok      "The model loaded on this host."
      POST /infer  (1x1 JPEG)      ok      12ms
                                           detections: 1
                                             fallen  0.91  bbox 0.10,0.20 0.30x0.40

Everything goes through the real ASGI app — the same routes, the same
body parsing, the same failure envelope — so what passes here is what
KAI-C will see. ``--image`` sends a real file instead of the built-in
1x1 frame; ``--param k=v`` adds caller params.
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

from opennvr_adapter_sdk.discovery import NotAnAdapter, asgi_app, identity, load

#: A valid 1x1 black JPEG — small enough to inline, real enough that a
#: decoder accepts it, so an image adapter gets past its first branch.
TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBg"
    "YFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoK"
    "CgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAA"
    "EDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIE"
    "AwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJi"
    "coKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWW"
    "l5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09f"
    "b3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQA"
    "AQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKj"
    "U2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJma"
    "oqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9"
    "oADAMBAAIRAxEAPwD+f+iiigD/2Q=="
)

#: A valid 8 kHz mono 16-bit WAV holding 100 samples of silence — the
#: audio equivalent of TINY_JPEG, so a Whisper-shaped adapter gets a
#: decodable body instead of an image field it never declared.
TINY_WAV = (
    b"RIFF" + (36 + 200).to_bytes(4, "little") + b"WAVEfmt "
    + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little")       # PCM
    + (1).to_bytes(2, "little")       # mono
    + (8000).to_bytes(4, "little")    # sample rate
    + (16000).to_bytes(4, "little")   # byte rate
    + (2).to_bytes(2, "little")       # block align
    + (16).to_bytes(2, "little")      # bits per sample
    + b"data" + (200).to_bytes(4, "little") + b"\x00" * 200
)

#: ``/infer`` JSON field per declared body shape. TEXT adapters take no
#: binary body at all.
_B64_FIELD: dict[str, str | None] = {
    "text": None,
    "image": "frame_b64",
    "audio": "audio_b64",
    "generic": "data_b64",
}

_SAMPLE_BODY: dict[str, bytes] = {
    "image": TINY_JPEG,
    "audio": TINY_WAV,
    "generic": b"opennvr-adapter-dev",
}

_SAMPLE_LABEL: dict[str, str] = {
    "image": "built-in 1x1 JPEG",
    "audio": "built-in silent WAV",
    "generic": "built-in 19-byte payload",
}

_OK = "ok"
_BAD = "FAIL"


def _line(verb: str, path: str, outcome: str, note: str = "") -> None:
    print(f"  {verb:4} {path:24} {outcome:7} {note}".rstrip())


def run_dev(
    directory: Path,
    *,
    image: str | None = None,
    params: dict[str, Any] | None = None,
    task: str | None = None,
    camera_id: str = "cam-1",
    repeat: int = 1,
) -> int:
    """Drive the adapter in ``directory`` through its own endpoints."""
    try:
        module, _name = load(Path(directory))
        app = asgi_app(module)
    except NotAnAdapter as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        from fastapi.testclient import TestClient
    except ImportError:  # pragma: no cover — fastapi is a hard dependency
        print("error: fastapi is required to run `dev`", file=sys.stderr)
        return 2

    name, version, tasks = identity(module)
    print(f"opennvr-adapter dev — {name} {version}")

    failures = 0

    started = time.monotonic()
    with TestClient(app) as client:
        print(f"  loading the model…{'':26}{_OK} "
              f"({time.monotonic() - started:.1f}s)\n")

        health = client.get("/health")
        ok = health.status_code == 200
        failures += not ok
        _line("GET", "/health", _OK if ok else _BAD,
              f"status={health.json().get('status')}" if ok
              else f"HTTP {health.status_code}")

        caps = client.get("/capabilities")
        ok = caps.status_code == 200
        failures += not ok
        if ok:
            payload = caps.json()
            advertised = payload.get("tasks_advertised") or list(tasks)
            _line("GET", "/capabilities", _OK,
                  "tasks: " + (", ".join(advertised) or "(none advertised!)"))
            model = payload.get("model") or {}
            if not model.get("fingerprint"):
                _line("", "", "warn",
                      "model.fingerprint is null — KAI-C skips drift detection "
                      "for this adapter")
        else:
            _line("GET", "/capabilities", _BAD, f"HTTP {caps.status_code}")

        hardware = client.get("/hardware/evaluation")
        ok = hardware.status_code == 200
        failures += not ok
        _line("GET", "/hardware/evaluation", _OK if ok else _BAD,
              f"{hardware.json().get('verdict')} — "
              f"{hardware.json().get('reasoning', '')}" if ok
              else f"HTTP {hardware.status_code}")

        metrics = client.get("/metrics")
        _line("GET", "/metrics", _OK if metrics.status_code == 200 else _BAD,
              f"{len(metrics.text.splitlines())} lines")

        shape = str(getattr(
            getattr(app, "state", None), "opennvr_body_shape", "image"))
        shape = getattr(shape, "value", shape).rsplit(".", 1)[-1].lower()
        if shape not in _B64_FIELD:
            shape = "image"
        field = _B64_FIELD[shape]

        request: dict[str, Any] = dict(params or {})
        request["camera_id"] = camera_id
        if task or tasks:
            request["task"] = task or tasks[0]

        if field is None:
            # A text adapter takes no binary body at all. Give it
            # something to say, or the run reports a successful call
            # that inferred on an empty string.
            if not any(k in request for k in ("text", "prompt", "input")):
                request["text"] = "Conformance check: hello, world."
                source = "built-in prompt"
            else:
                source = "your text param"
            if image is not None:
                print("  note: --image is ignored; this adapter declares no "
                      "binary body.")
        elif image is not None:
            body = Path(image).expanduser().read_bytes()
            source = Path(image).name
            request[field] = base64.b64encode(body).decode()
        else:
            body = _SAMPLE_BODY[shape]
            source = _SAMPLE_LABEL[shape]
            request[field] = base64.b64encode(body).decode()

        for attempt in range(max(1, repeat)):
            response = client.post("/infer", json=request)
            note = f" ({source})" if attempt == 0 else ""
            if response.status_code == 200:
                payload = response.json()
                _line("POST", f"/infer{note}", _OK,
                      f"{payload.get('inference_ms', 0)}ms")
                if attempt == 0:
                    _describe(payload.get("result") or {})
            else:
                failures += 1
                error = (response.json().get("error") or {})
                _line("POST", f"/infer{note}", _BAD,
                      f"HTTP {response.status_code} {error.get('category', '')} "
                      f"{error.get('code', '')}")
                print(f"        {error.get('message', response.text[:200])}")
                break

    print()
    if failures:
        print(f"  {failures} problem(s). `opennvr-adapter validate .` explains "
              f"what the contract expects.")
        return 1
    print("  All green. Next: `opennvr-adapter validate .` for the full "
          "conformance run.")
    return 0


def _describe(result: dict[str, Any]) -> None:
    """Show the model's answer in the shape a consumer will read it."""
    detections = result.get("detections")
    if isinstance(detections, list):
        print(f"        detections: {len(detections)}")
        for item in detections[:5]:
            if not isinstance(item, dict):
                continue
            box = item.get("bbox") or {}
            print(f"          {str(item.get('label', '?')):16}"
                  f"{float(item.get('confidence', 0)):.2f}  "
                  f"bbox {box.get('x', 0):.2f},{box.get('y', 0):.2f} "
                  f"{box.get('w', 0):.2f}x{box.get('h', 0):.2f}")
        if not detections:
            print("          (none — expected, for a 1x1 frame)")
        return
    rendered = json.dumps(result)[:300]
    print(f"        result: {rendered}")


__all__ = ["run_dev", "TINY_JPEG", "TINY_WAV"]

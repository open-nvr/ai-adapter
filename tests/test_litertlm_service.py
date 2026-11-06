# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LitertLmService: routing (caption vs VQA), the image downscale, readiness
and errors, the one-time model download and the pinned Qualcomm library
fetch — with the LiteRT-LM engine and the network stubbed. The real NPU run
was measured on a Ventuno Q (see the adapter README)."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from opennvr_adapter_sdk import ErrorCategory, HealthStatus, ServiceError

from adapters.litertlm import service as svc_mod
from adapters.litertlm.service import (
    QAIRT_FILES,
    QAIRT_ROOT,
    LitertLmService,
    download_model,
    fetch_qairt_libraries,
    shrink_image,
)


def _jpeg(w: int = 32, h: int = 32) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeConversation:
    def __init__(self, log):
        self.log = log
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def send_message(self, message):
        self.log.append(message)
        text = message["content"][-1]["text"]
        return {"content": [{"type": "text", "text": f"  echo: {text}  "}]}


class _FakeEngine:
    def __init__(self):
        self.log = []
        self.kwargs = None
    def create_conversation(self, **kwargs):
        self.kwargs = kwargs
        return _FakeConversation(self.log)


def _ready(tmp_path, **kw) -> tuple[LitertLmService, _FakeEngine]:
    engine = _FakeEngine()
    model = tmp_path / "gemma.litertlm"
    model.write_bytes(b"m" * 10)
    s = LitertLmService(model_path=str(model), model_url="", lib_dir=tmp_path / "lib",
                        runtime_dir=tmp_path / "rt", fetch_libraries=False,
                        engine_factory=lambda: engine, **kw)
    s._engine = engine
    s._state = HealthStatus.OK
    s._fingerprint_cache = s._compute_fingerprint()
    return s, engine


def test_caption_by_default_and_vqa_with_a_question(tmp_path):
    s, engine = _ready(tmp_path)
    r = s.infer({"__file__": _jpeg()})
    assert r.result["task"] == "scene_caption" and r.result["caption"].startswith("echo:")
    assert r.result["accelerator"] == "qualcomm-hexagon-npu" and r.inference_ms >= 0
    assert engine.log[-1]["content"][0]["type"] == "image" and engine.log[-1]["content"][0]["path"].endswith(".jpg")
    r = s.infer({"__file__": _jpeg(), "question": "what colour is the shirt?"})
    assert r.result["task"] == "visual_qa" and r.result["answer"] == "echo: what colour is the shirt?"
    r = s.infer({"__file__": _jpeg(), "task": "scene_caption", "question": "ignored"})
    assert "caption" in r.result and "answer" not in r.result
    assert engine.kwargs["max_output_tokens"] == svc_mod.MAX_OUTPUT_TOKENS


def test_the_photo_is_downscaled_before_the_model(tmp_path):
    s, engine = _ready(tmp_path, max_image_px=200)
    r = s.infer({"__file__": _jpeg(1920, 1080)})
    assert r.result["image_px"] == [200, 112]
    small, size = shrink_image(_jpeg(100, 50), 768)
    assert size == (100, 50), "never upscaled"


def test_errors_missing_image_bad_image_unknown_task_not_ready(tmp_path):
    s, _ = _ready(tmp_path)
    with pytest.raises(ServiceError) as e:
        s.infer({})
    assert e.value.http_status == 400 and e.value.code == "missing_image"
    with pytest.raises(ServiceError) as e:
        s.infer({"__file__": b"not an image"})
    assert e.value.code == "invalid_image"
    with pytest.raises(ServiceError) as e:
        s.infer({"__file__": _jpeg(), "task": "ocr"})
    assert e.value.category == ErrorCategory.NOT_SUPPORTED
    s._state = HealthStatus.LOADING
    with pytest.raises(ServiceError) as e:
        s.infer({"__file__": _jpeg()})
    assert e.value.http_status == 503 and e.value.transient
    s._state, s._error = HealthStatus.ERROR, "no NPU"
    with pytest.raises(ServiceError) as e:
        s.infer({"__file__": _jpeg()})
    assert not e.value.transient and "no NPU" in e.value.message


def test_health_and_hardware_follow_the_load_state(tmp_path, monkeypatch):
    s, _ = _ready(tmp_path)
    assert s.is_ready() and s.health_status() == HealthStatus.OK
    assert s.hardware_evaluation().verdict.value == "ok"
    assert s.model_info().framework == "litert-lm" and s.fingerprint().startswith("sha256-id:")
    s._state = HealthStatus.LOADING
    assert s.hardware_evaluation().verdict.value == "warn"
    # no NPU node on this machine: the load fails with a clear reason, the service stays up
    monkeypatch.setattr(svc_mod.os.path, "exists", lambda p: False if p == svc_mod.FASTRPC_NODE else Path(p).exists())
    t = LitertLmService(model_path=str(tmp_path / "x.litertlm"), model_url="", fetch_libraries=False,
                        engine_factory=lambda: None)
    t.load_now()
    assert t.health_status() == HealthStatus.ERROR and "fastrpc" in (t._error or "")
    assert t.hardware_evaluation().verdict.value == "blocked"


def test_model_is_downloaded_once_and_verified(tmp_path):
    body = b"g" * 100
    sha = hashlib.sha256(body).hexdigest()
    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def opener(url, timeout=0):
        calls.append(url)
        return _Resp(body)

    dest = tmp_path / "m" / "gemma.litertlm"
    download_model("http://x/m", dest, sha, opener=opener)
    assert dest.read_bytes() == body and calls == ["http://x/m"] and not dest.with_suffix(".litertlm.part").exists()
    with pytest.raises(OSError):
        download_model("http://x/m", tmp_path / "bad.litertlm", "0" * 64, opener=opener)
    assert not (tmp_path / "bad.litertlm").exists()


def test_qualcomm_libraries_are_read_out_of_the_zip_by_range_and_checked(tmp_path, monkeypatch):
    """A small fake SDK zip served through range requests; the pinned digests
    are swapped for the fake content's so the check itself is exercised."""
    buf = io.BytesIO()
    content = {name: f"lib {name}".encode() for name in QAIRT_FILES}
    with zipfile.ZipFile(buf, "w") as z:
        for name, (folder, _) in QAIRT_FILES.items():
            z.writestr(QAIRT_ROOT + folder + name, content[name])
    blob = buf.getvalue()
    monkeypatch.setattr(svc_mod, "QAIRT_FILES", {n: (f, hashlib.sha256(content[n]).hexdigest()) for n, (f, _) in QAIRT_FILES.items()})

    class _Resp:
        def __init__(self, data):
            self.data, self.status = data, 206
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return self.data

    def opener(req, timeout=0):
        rng = req.headers["Range"].split("=")[1]
        a, b = (int(x) for x in rng.split("-"))
        return _Resp(blob[a:b + 1])

    lib = tmp_path / "lib"
    fetched = fetch_qairt_libraries(lib, opener=opener, url="http://x/sdk.zip", size=len(blob))
    assert sorted(fetched) == sorted(QAIRT_FILES) and (lib / "libQnnHtp.so").read_bytes() == b"lib libQnnHtp.so"
    assert fetch_qairt_libraries(lib, opener=opener, url="http://x/sdk.zip", size=len(blob)) == [], "all present: nothing fetched"
    (lib / "libQnnSystem.so").write_bytes(b"corrupt")
    assert fetch_qairt_libraries(lib, opener=opener, url="http://x/sdk.zip", size=len(blob)) == ["libQnnSystem.so"]


def test_load_order_node_libraries_model_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(svc_mod.os.path, "exists", lambda p: True if p == svc_mod.FASTRPC_NODE else Path(p).exists())
    order = []
    monkeypatch.setattr(svc_mod, "fetch_qairt_libraries", lambda lib_dir, **k: order.append("libs") or [])
    monkeypatch.setattr(svc_mod, "download_model", lambda url, dest, sha: (order.append("model"), dest.write_bytes(b"m")))
    engine = _FakeEngine()
    s = LitertLmService(model_path=str(tmp_path / "gemma.litertlm"), model_url="http://x/m",
                        lib_dir=tmp_path / "lib", engine_factory=lambda: order.append("engine") or engine)
    s.load_now()
    assert order == ["libs", "model", "engine"] and s.is_ready()
    assert s.infer({"__file__": _jpeg()}).result["caption"].startswith("echo:")

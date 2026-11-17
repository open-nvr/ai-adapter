# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The InsightFace adapter installs its model pack itself and retries a
failed download in the background.

Before this, a pack download that failed at boot (a reset TLS handshake
on a flaky uplink) left the adapter answering ``model_not_loaded`` with
``transient: false`` until someone restarted the container, while
``/health`` said ``loading`` and the Docker healthcheck stayed green.

Downloads here use ``file://`` URLs, so no test touches the network.
"""
from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path

import pytest

from opennvr_adapter_sdk import ErrorCategory, HealthStatus, ServiceError
from tests._insightface_service_fixtures import (  # noqa: F401
    insightface_environment,
    sample_jpeg,
)

_PACK_FILES = {"det_10g.onnx": b"detector", "w600k_r50.onnx": b"recogniser"}


def _service_module():
    if "adapters.insightface.service" in sys.modules:
        importlib.reload(sys.modules["adapters.insightface.service"])
    import adapters.insightface.service as service_module
    return service_module


def _write_zip(path: Path, files: dict[str, bytes], prefix: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(prefix + name, data)
    return path


def _url_for(source_dir: Path) -> str:
    # ``{pack}`` is filled in by ensure_model_pack, as with the real URL.
    return source_dir.resolve().as_uri() + "/{pack}.zip"


def _pack_dir(root: Path, pack: str = "buffalo_l") -> Path:
    return root / "models" / pack


# ── ensure_model_pack ──────────────────────────────────────────────


def test_installs_a_flat_pack_and_leaves_nothing_behind(tmp_path):
    mod = _service_module()
    src = tmp_path / "mirror"
    _write_zip(src / "buffalo_l.zip", _PACK_FILES)
    root = tmp_path / "cache"

    got = mod.ensure_model_pack(str(root), "buffalo_l", _url_for(src))

    pack = _pack_dir(root)
    assert Path(got) == pack
    assert sorted(p.name for p in pack.iterdir()) == sorted(_PACK_FILES)
    assert sorted(p.name for p in (root / "models").iterdir()) == ["buffalo_l"]


def test_flattens_a_pack_zipped_inside_its_own_folder(tmp_path):
    mod = _service_module()
    src = tmp_path / "mirror"
    _write_zip(src / "antelopev2.zip", _PACK_FILES, prefix="antelopev2/")
    root = tmp_path / "cache"

    mod.ensure_model_pack(str(root), "antelopev2", _url_for(src))

    pack = _pack_dir(root, "antelopev2")
    assert sorted(p.name for p in pack.iterdir()) == sorted(_PACK_FILES)
    assert sorted(p.name for p in (root / "models").iterdir()) == ["antelopev2"]


def test_an_installed_pack_is_never_fetched(tmp_path):
    mod = _service_module()
    root = tmp_path / "cache"
    pack = _pack_dir(root)
    pack.mkdir(parents=True)
    (pack / "mine.onnx").write_bytes(b"hand-copied")

    mod.ensure_model_pack(str(root), "buffalo_l", _url_for(tmp_path / "nowhere"))

    assert [p.name for p in pack.iterdir()] == ["mine.onnx"]


def test_replaces_the_empty_directory_a_cut_download_left(tmp_path):
    mod = _service_module()
    src = tmp_path / "mirror"
    _write_zip(src / "buffalo_l.zip", _PACK_FILES)
    root = tmp_path / "cache"
    _pack_dir(root).mkdir(parents=True)  # what InsightFace leaves

    mod.ensure_model_pack(str(root), "buffalo_l", _url_for(src))

    assert sorted(p.name for p in _pack_dir(root).iterdir()) == sorted(_PACK_FILES)


def test_drops_a_truncated_leftover_zip_and_fetches_again(tmp_path):
    mod = _service_module()
    src = tmp_path / "mirror"
    good = _write_zip(src / "buffalo_l.zip", _PACK_FILES)
    root = tmp_path / "cache"
    leftover = root / "models" / "buffalo_l.zip"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(good.read_bytes()[:40])  # cut short

    mod.ensure_model_pack(str(root), "buffalo_l", _url_for(src))

    assert sorted(p.name for p in _pack_dir(root).iterdir()) == sorted(_PACK_FILES)
    assert not leftover.exists()


def test_a_corrupt_download_leaves_nothing_a_retry_would_trust(tmp_path):
    mod = _service_module()
    src = tmp_path / "mirror"
    src.mkdir()
    (src / "buffalo_l.zip").write_bytes(b"this is not a zip")
    root = tmp_path / "cache"

    with pytest.raises(zipfile.BadZipFile):
        mod.ensure_model_pack(str(root), "buffalo_l", _url_for(src))

    assert sorted(p.name for p in (root / "models").iterdir()) == []


def test_a_zip_without_onnx_files_is_refused(tmp_path):
    mod = _service_module()
    src = tmp_path / "mirror"
    _write_zip(src / "buffalo_l.zip", {"README.txt": b"no models"})
    root = tmp_path / "cache"

    with pytest.raises(zipfile.BadZipFile):
        mod.ensure_model_pack(str(root), "buffalo_l", _url_for(src))

    assert sorted(p.name for p in (root / "models").iterdir()) == []


def test_missing_pack_with_fetching_off_is_a_clear_error(tmp_path):
    mod = _service_module()
    with pytest.raises(FileNotFoundError, match="OPENNVR_INSIGHTFACE_MODEL_URL"):
        mod.ensure_model_pack(str(tmp_path / "cache"), "buffalo_l", "")


# ── Service: retry and honest health ───────────────────────────────


def _service(env, monkeypatch, *, model_root: Path, url: str):
    monkeypatch.setenv("OPENNVR_INSIGHTFACE_MODEL_ROOT", str(model_root))
    monkeypatch.setenv("OPENNVR_INSIGHTFACE_MODEL_URL", url)
    mod = _service_module()
    from adapters.insightface.face_db import FaceDB

    svc = mod.InsightFaceService(face_db=FaceDB(storage_path=str(env["db_path"])))
    timers: list[float] = []
    svc._start_retry_timer = lambda delay: timers.append(delay)  # fire by hand
    return mod, svc, timers


def test_a_failed_download_is_retried_until_it_succeeds(
    insightface_environment, monkeypatch, tmp_path, sample_jpeg,
):
    mirror = tmp_path / "mirror"
    root = tmp_path / "cache"
    _mod, svc, timers = _service(
        insightface_environment, monkeypatch, model_root=root, url=_url_for(mirror))

    svc.load()  # the mirror has no zip yet: the download fails

    assert not svc.is_ready()
    assert svc.health_status() == HealthStatus.LOADING
    assert timers == [5.0]
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "face_detection"})
    err = excinfo.value
    assert err.category == ErrorCategory.MODEL_ERROR
    assert err.code == "model_not_loaded"
    assert err.transient is True
    assert err.retry_after_ms is not None and err.retry_after_ms >= 1000
    assert "retrying in 5s" in str(err.message)

    _write_zip(mirror / "buffalo_l.zip", _PACK_FILES)  # the link comes back
    svc.load()  # what the timer would run

    assert svc.is_ready()
    assert svc.health_status() == HealthStatus.OK
    assert timers == [5.0]
    out = svc.infer({"__file__": sample_jpeg, "task": "face_detection"})
    assert out.result["face_count"] == 1


def test_retry_delay_doubles_up_to_the_cap(insightface_environment, monkeypatch, tmp_path):
    mod, svc, timers = _service(
        insightface_environment, monkeypatch,
        model_root=tmp_path / "cache", url=_url_for(tmp_path / "nowhere"))

    for _ in range(9):
        svc.load()

    assert timers == [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0, 300.0]
    assert timers[-1] == mod.RETRY_MAX_DELAY_S


def test_missing_pack_with_fetching_off_reports_error_not_loading(
    insightface_environment, monkeypatch, tmp_path, sample_jpeg,
):
    _mod, svc, timers = _service(
        insightface_environment, monkeypatch, model_root=tmp_path / "empty", url="")

    svc.load()

    assert svc.health_status() == HealthStatus.ERROR
    assert timers == []
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "face_detection"})
    assert excinfo.value.transient is False
    assert excinfo.value.retry_after_ms is None


def test_a_model_that_fails_to_build_reports_error(insightface_environment, monkeypatch):
    svc = _service(
        insightface_environment, monkeypatch,
        model_root=insightface_environment["model_root"], url="")[1]

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated onnxruntime failure")

    monkeypatch.setattr(sys.modules["insightface.app"], "FaceAnalysis", _Boom)
    svc.load()

    assert svc.health_status() == HealthStatus.ERROR


def test_face_analysis_is_pointed_at_the_configured_cache(insightface_environment, monkeypatch):
    root = insightface_environment["model_root"]
    svc = _service(insightface_environment, monkeypatch, model_root=root, url="")[1]

    svc.load()

    assert svc._face_app.init_kwargs["root"] == str(root)


def test_health_endpoint_says_error_for_a_failed_load(insightface_environment, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    monkeypatch.setenv("OPENNVR_INSIGHTFACE_MODEL_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("OPENNVR_INSIGHTFACE_MODEL_URL", "")
    for name in ("adapters.insightface.service", "adapters.insightface.main"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    from fastapi.testclient import TestClient
    import adapters.insightface.main as main_module

    with TestClient(main_module.app) as client:
        body = client.get("/health").json()

    assert body["status"] == "error"

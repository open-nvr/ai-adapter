# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""opennvr_adapter_sdk.providers — the one provider resolver every ONNX adapter uses."""
from __future__ import annotations

import logging

import pytest

from opennvr_adapter_sdk.providers import (
    accelerator_available,
    describe_providers,
    effective_providers,
    parse_provider_specs,
    requested_providers,
    resolve_providers,
    session_providers,
)


def test_parse_shorthand_and_multi_device_continuation():
    assert parse_provider_specs("OpenVINOExecutionProvider:NPU,CPUExecutionProvider") == [
        ("OpenVINOExecutionProvider", "NPU"), ("CPUExecutionProvider", None)]
    assert parse_provider_specs("OpenVINOExecutionProvider:AUTO:NPU,GPU,CPUExecutionProvider") == [
        ("OpenVINOExecutionProvider", "AUTO:NPU,GPU"), ("CPUExecutionProvider", None)]
    assert parse_provider_specs(["CUDAExecutionProvider:1"]) == [("CUDAExecutionProvider", "1")]
    assert parse_provider_specs([("OpenVINOExecutionProvider", {"device_type": "GPU"})]) == [
        ("OpenVINOExecutionProvider", "GPU")]
    assert parse_provider_specs(None) == [] and parse_provider_specs("") == []


def test_resolve_carries_device_options_and_keeps_cpu_last():
    provs = resolve_providers("OpenVINOExecutionProvider:NPU",
                              available=["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert provs == [("OpenVINOExecutionProvider", {"device_type": "NPU", "disable_dynamic_shapes": "True"}),
                     "CPUExecutionProvider"]
    assert resolve_providers("CUDAExecutionProvider:1", available=["CUDAExecutionProvider"]) == [
        ("CUDAExecutionProvider", {"device_id": "1"}), "CPUExecutionProvider"]


def test_resolve_skips_a_provider_this_build_lacks_and_never_fails():
    provs = resolve_providers("OpenVINOExecutionProvider:NPU,CUDAExecutionProvider",
                              available=["CPUExecutionProvider"])
    assert provs == ["CPUExecutionProvider"]
    assert resolve_providers(None, ["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert resolve_providers("CPUExecutionProvider:0", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_describe_is_human_readable():
    assert describe_providers([("OpenVINOExecutionProvider", {"device_type": "NPU"}),
                               "CPUExecutionProvider"]) == \
        "OpenVINOExecutionProvider(NPU), CPUExecutionProvider"
    assert describe_providers(None) == ""


def test_env_setting_and_default(monkeypatch):
    monkeypatch.delenv("ADAPTER_ONNX_PROVIDERS", raising=False)
    assert requested_providers() is None
    assert requested_providers("CUDAExecutionProvider") == "CUDAExecutionProvider"
    monkeypatch.setenv("ADAPTER_ONNX_PROVIDERS", " OpenVINOExecutionProvider:GPU ")
    assert requested_providers("x") == "OpenVINOExecutionProvider:GPU"


def test_accelerator_available_ignores_cpu_and_azure():
    assert accelerator_available(["CPUExecutionProvider"]) is False
    assert accelerator_available(["AzureExecutionProvider", "CPUExecutionProvider"]) is False
    assert accelerator_available(["OpenVINOExecutionProvider", "CPUExecutionProvider"]) is True
    assert accelerator_available(["CUDAExecutionProvider", "CPUExecutionProvider"]) is True


class _Sess:
    def __init__(self, provs): self._p = provs
    def get_providers(self): return list(self._p)


def test_effective_providers_warns_on_silent_fallback(caplog):
    req = [("OpenVINOExecutionProvider", {"device_type": "NPU"}), "CPUExecutionProvider"]
    with caplog.at_level(logging.WARNING, logger="opennvr_adapter_sdk.providers"):
        info = effective_providers(_Sess(["CPUExecutionProvider"]), req, adapter="yolov8")
    assert info["effective_providers"] == ["CPUExecutionProvider"]
    assert info["accelerator_in_use"] is False and info["accelerator_requested"] is True
    assert info["requested_providers"] == "OpenVINOExecutionProvider(NPU), CPUExecutionProvider"
    assert "yolov8: ONNX Runtime could not use OpenVINOExecutionProvider(NPU)" in caplog.text


def test_effective_providers_reports_accelerator_in_use(caplog):
    req = [("OpenVINOExecutionProvider", {"device_type": "NPU"}), "CPUExecutionProvider"]
    with caplog.at_level(logging.WARNING, logger="opennvr_adapter_sdk.providers"):
        info = effective_providers(_Sess(["OpenVINOExecutionProvider", "CPUExecutionProvider"]), req)
    assert info["accelerator_in_use"] is True
    assert "could not use" not in caplog.text


def test_session_providers_uses_env_against_this_build(monkeypatch):
    import opennvr_adapter_sdk.providers as m
    monkeypatch.setattr(m, "available_providers", lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setenv("ADAPTER_ONNX_PROVIDERS", "OpenVINOExecutionProvider:GPU")
    assert session_providers() == [
        ("OpenVINOExecutionProvider", {"device_type": "GPU", "disable_dynamic_shapes": "True"}),
        "CPUExecutionProvider"]
    monkeypatch.delenv("ADAPTER_ONNX_PROVIDERS")
    assert session_providers() == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]  # ORT's order
    assert session_providers("CUDAExecutionProvider") == ["CPUExecutionProvider"]        # not in this build


def test_openvino_accelerator_devices_compile_static(monkeypatch):
    from opennvr_adapter_sdk.providers import DYNAMIC_SHAPES_ENV, static_shapes, wants_static_shapes

    monkeypatch.delenv(DYNAMIC_SHAPES_ENV, raising=False)
    gpu = resolve_providers("OpenVINOExecutionProvider:GPU", ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert gpu[0] == ("OpenVINOExecutionProvider", {"device_type": "GPU", "disable_dynamic_shapes": "True"})
    assert wants_static_shapes(gpu)
    cpu = resolve_providers("OpenVINOExecutionProvider:CPU", ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert cpu[0] == ("OpenVINOExecutionProvider", {"device_type": "CPU"})
    assert not wants_static_shapes(cpu)
    assert not wants_static_shapes(["CPUExecutionProvider"])
    monkeypatch.setenv(DYNAMIC_SHAPES_ENV, "1")
    npu = resolve_providers("OpenVINOExecutionProvider:NPU", ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert npu[0] == ("OpenVINOExecutionProvider", {"device_type": "NPU"})
    # pinning the shape replaces the first-inference compile with one reshape
    pinned = static_shapes(gpu, {"images": [1, 3, 448, 448]})
    assert pinned == [
        ("OpenVINOExecutionProvider", {"device_type": "GPU", "reshape_input": "images[1,3,448,448]"}),
        "CPUExecutionProvider",
    ]
    assert gpu[0][1] == {"device_type": "GPU", "disable_dynamic_shapes": "True"}  # input untouched


def test_build_with_fallback_rebuilds_on_cpu_and_reports(caplog):
    from opennvr_adapter_sdk.providers import build_with_fallback

    calls = []

    def build(providers):
        calls.append(list(providers))
        if providers[0] != "CPUExecutionProvider":
            raise RuntimeError("no OpenCL device")
        return "cpu-session"

    gpu = [("OpenVINOExecutionProvider", {"device_type": "GPU"}), "CPUExecutionProvider"]
    with caplog.at_level(logging.ERROR, logger="opennvr_adapter_sdk.providers"):
        obj, used, err = build_with_fallback(build, gpu, adapter="t")
    assert obj == "cpu-session" and used == ["CPUExecutionProvider"]
    assert err.startswith("RuntimeError: no OpenCL device")
    assert calls == [gpu, ["CPUExecutionProvider"]]
    assert "could not start on OpenVINOExecutionProvider(GPU)" in caplog.text

    class Sess:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    report = effective_providers(Sess(), used, adapter="t", fallback_error=err)
    assert report["accelerator_in_use"] is False
    assert report["accelerator_requested"] is True
    assert report["accelerator_fallback_error"] == err


def test_build_with_fallback_propagates_cpu_failures():
    from opennvr_adapter_sdk.providers import build_with_fallback

    def build(_providers):
        raise ValueError("bad model")

    with pytest.raises(ValueError):
        build_with_fallback(build, ["CPUExecutionProvider"], adapter="t")
    assert build_with_fallback(lambda p: "ok", None)[1] == ["CPUExecutionProvider"]


# ── plugin execution providers (QNN, the qualcomm flavour) ───────────────

def _reset_plugins(monkeypatch):
    import opennvr_adapter_sdk.providers as p
    monkeypatch.setattr(p, "_registered_plugin_eps", set())
    monkeypatch.setattr(p, "_ORIGINAL_SESSION", None)
    return p


class _Dev:
    def __init__(self, ep_name):
        self.ep_name = ep_name


class _Opts:
    def __init__(self):
        self.added = []
        self.config = {}

    def add_provider(self, name, opts):          # by name: refused for plugins by ORT
        raise AssertionError("plugin providers must be attached by device")

    def add_provider_for_devices(self, devices, opts):
        self.added.append((devices[0].ep_name, dict(opts)))

    def add_session_config_entry(self, k, v):
        self.config[k] = v


def _fake_ort(monkeypatch, *, devices=("QNNExecutionProvider",), compiles=True, record=None):
    """A fake onnxruntime module whose InferenceSession dumps the compiled
    graph where the session options say (as the real runtime does)."""
    import sys
    import types

    calls = record if record is not None else {}

    class _Session:
        def __init__(self, model, sess_options=None, providers=None, provider_options=None, **kw):
            calls.setdefault("models", []).append(model)
            calls["providers"] = providers
            calls["opts"] = sess_options
            dump = (getattr(sess_options, "config", {}) or {}).get("ep.context_file_path")
            if dump and compiles:
                open(dump, "wb").write(b"ctx")

        def get_providers(self):
            return ["QNNExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            return []

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = _Session
    fake.SessionOptions = _Opts
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake.register_execution_provider_library = lambda name, path: calls.setdefault("registered", []).append(name)
    fake.get_ep_devices = lambda: [_Dev(d) for d in devices]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", types.SimpleNamespace(
        get_library_path=lambda: "/x/libonnxruntime_providers_qnn.so",
        get_qnn_htp_path=lambda: "/x/libQnnHtp.so", get_qnn_cpu_path=lambda: "/x/libQnnCpu.so"))
    return fake, calls


def test_qnn_shorthand_is_the_backend_and_options_layer(monkeypatch):
    from opennvr_adapter_sdk.providers import provider_options
    _reset_plugins(monkeypatch)
    _fake_ort(monkeypatch)
    provs = resolve_providers("QNNExecutionProvider:htp", ["QNNExecutionProvider", "CPUExecutionProvider"])
    assert provs == [("QNNExecutionProvider", {"backend_type": "htp"}), "CPUExecutionProvider"]
    opts = provider_options(provs[0], env={})
    assert opts["backend_path"] == "/x/libQnnHtp.so" and "backend_type" not in opts
    assert opts["enable_htp_fp16_precision"] == "1"
    assert opts["htp_performance_mode"] == "sustained_high_performance"
    assert provider_options(("QNNExecutionProvider", {"backend_type": "cpu"}), env={})["backend_path"] == "/x/libQnnCpu.so"
    over = provider_options(provs[0], env={"ADAPTER_ONNX_PROVIDER_OPTIONS": "htp_performance_mode=burst, vtcm_mb=8"})
    assert over["htp_performance_mode"] == "burst" and over["vtcm_mb"] == "8"
    assert provider_options(("CUDAExecutionProvider", {"device_id": "1"}), env={}) == {"device_id": "1"}


def test_session_providers_registers_the_plugin_once_and_installs_the_shim(monkeypatch, caplog):
    p = _reset_plugins(monkeypatch)
    fake, calls = _fake_ort(monkeypatch)
    monkeypatch.setenv("ADAPTER_ONNX_PROVIDERS", "QNNExecutionProvider:htp")
    assert session_providers()[0] == ("QNNExecutionProvider", {"backend_type": "htp"})
    assert session_providers()[0][0] == "QNNExecutionProvider"
    assert calls["registered"] == ["QNNExecutionProvider"]            # once
    assert fake.InferenceSession.__name__ == "InferenceSession" and hasattr(fake.InferenceSession, "__wrapped__")
    # a build without the package: the request degrades to CPU, said out loud
    p2 = _reset_plugins(monkeypatch)
    import sys
    monkeypatch.delitem(sys.modules, "onnxruntime_qnn")
    monkeypatch.setattr("importlib.import_module", lambda name: (_ for _ in ()).throw(ImportError(name)))
    with caplog.at_level(logging.WARNING):
        assert session_providers("QNNExecutionProvider:htp") == ["CPUExecutionProvider"]
    assert "needs the onnxruntime_qnn package" in caplog.text


def test_shim_routes_a_library_providers_list_through_the_plugin_api(monkeypatch, tmp_path):
    """InsightFace / fast-plate-ocr / onnx-asr call ort.InferenceSession(path,
    providers=[...]) themselves; the shim attaches the plugin by device,
    caches the compiled graph next to the model, and loads it next time."""
    from opennvr_adapter_sdk.providers import effective_providers
    _reset_plugins(monkeypatch)
    fake, calls = _fake_ort(monkeypatch)
    model = tmp_path / "det.onnx"; model.write_bytes(b"m")
    provs = session_providers("QNNExecutionProvider:htp")
    sess = fake.InferenceSession(str(model), providers=provs)       # what the library does
    assert calls["providers"] is None                                # the plugin never reaches ORT's list
    assert calls["opts"].added == [("QNNExecutionProvider", {
        "backend_path": "/x/libQnnHtp.so", "htp_performance_mode": "sustained_high_performance",
        "enable_htp_fp16_precision": "1", "htp_graph_finalization_optimization_mode": "3"})]
    ctx = tmp_path / "det.onnx.QNNExecutionProvider.ctx.onnx"
    assert ctx.exists() and sess._opennvr_plugin_compiled is True
    assert effective_providers(sess, provs, adapter="t")["accelerator_in_use"] is True
    # second session: the cache is the model now
    fake.InferenceSession(str(model), providers=provs)
    assert calls["models"][-1] == str(ctx)
    # the parallel provider_options form (fast-plate-ocr) and a plain CPU list pass through
    fake.InferenceSession(str(model), providers=["QNNExecutionProvider", "CPUExecutionProvider"],
                          provider_options=[{"vtcm_mb": "4"}, {}])
    assert calls["opts"].added[-1][1]["vtcm_mb"] == "4"
    fake.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    assert calls["providers"] == ["CPUExecutionProvider"]


def test_plugin_that_compiled_nothing_is_reported_as_cpu(monkeypatch, tmp_path, caplog):
    from opennvr_adapter_sdk.providers import effective_providers
    _reset_plugins(monkeypatch)
    fake, calls = _fake_ort(monkeypatch, compiles=False)
    model = tmp_path / "m.onnx"; model.write_bytes(b"m")
    provs = session_providers("QNNExecutionProvider:htp")
    sess = fake.InferenceSession(str(model), providers=provs)
    assert sess._opennvr_plugin_compiled is False
    with caplog.at_level(logging.WARNING):
        report = effective_providers(sess, provs, adapter="t")
    assert report["effective_providers"] == ["CPUExecutionProvider"]
    assert report["accelerator_in_use"] is False and report["accelerator_requested"] is True
    assert "could not use QNNExecutionProvider(htp)" in caplog.text
    assert not (tmp_path / "m.onnx.QNNExecutionProvider.ctx.onnx").exists()


def test_plugin_without_a_device_builds_the_classic_session(monkeypatch, tmp_path, caplog):
    _reset_plugins(monkeypatch)
    fake, calls = _fake_ort(monkeypatch, devices=())
    model = tmp_path / "m.onnx"; model.write_bytes(b"m")
    provs = session_providers("QNNExecutionProvider:htp")
    with caplog.at_level(logging.WARNING):
        sess = fake.InferenceSession(str(model), providers=provs)
    assert calls["providers"] == ["CPUExecutionProvider"] and sess._opennvr_plugin_compiled is None
    assert "reports no device" in caplog.text


def test_qnn_wants_static_shapes_and_pins_through_a_fixed_copy(monkeypatch, tmp_path):
    """QNN compiles for fixed shapes only: static_shapes() leaves the spec on
    the entry and plugin_session writes a pinned copy of the model first."""
    import opennvr_adapter_sdk.providers as p
    from opennvr_adapter_sdk.providers import static_shapes, wants_static_shapes
    _reset_plugins(monkeypatch)
    fake, calls = _fake_ort(monkeypatch)
    provs = session_providers("QNNExecutionProvider:htp")
    assert wants_static_shapes(provs) is True
    assert wants_static_shapes(resolve_providers("QNNExecutionProvider:cpu", ["QNNExecutionProvider"])) is False
    pinned = static_shapes(provs, {"images": [1, 3, 448, 448]})
    assert pinned[0][1][p.FIXED_SHAPES_KEY] == "images[1,3,448,448]" and pinned[1] == "CPUExecutionProvider"
    model = tmp_path / "pose.onnx"; model.write_bytes(b"dyn")
    seen = {}
    fixed = str(tmp_path / "pose.onnx.QNNExecutionProvider.fixed.onnx")

    def _pin(m, spec, prov, adapter="adapter"):
        seen["call"] = (m, spec, prov)
        return fixed
    monkeypatch.setattr(p, "fixed_shape_model", _pin)
    (tmp_path / "pose.onnx.QNNExecutionProvider.fixed.onnx").write_bytes(b"fixed")
    fake.InferenceSession(str(model), providers=pinned)
    assert seen["call"] == (str(model), "images[1,3,448,448]", "QNNExecutionProvider")
    assert calls["models"][-1].endswith("pose.onnx.QNNExecutionProvider.fixed.onnx")
    assert p.FIXED_SHAPES_KEY not in calls["opts"].added[-1][1]        # never handed to the runtime


def test_fixed_shape_model_without_onnx_keeps_the_dynamic_model(monkeypatch, tmp_path, caplog):
    import sys
    from opennvr_adapter_sdk.providers import fixed_shape_model
    model = tmp_path / "m.onnx"; model.write_bytes(b"dyn")
    monkeypatch.setitem(sys.modules, "onnx", None)                # import fails
    with caplog.at_level(logging.WARNING):
        out = fixed_shape_model(str(model), "images[1,3,448,448]", "QNNExecutionProvider", adapter="t")
    assert out == str(model) and "onnx package is not installed" in caplog.text
    assert fixed_shape_model(b"bytes", "x[1]", "QNNExecutionProvider") == b"bytes"


def test_fixed_shape_model_wildcard_pins_only_a_dynamic_first_input(monkeypatch, tmp_path):
    """InsightFace builds five sessions from one provider list: '*' pins the
    dynamic detector and leaves the fixed-size models alone."""
    import sys
    import types
    from opennvr_adapter_sdk.providers import fixed_shape_model

    class _Dim:
        def __init__(self, v=None, p=""):
            self.dim_value = v or 0; self.dim_param = p

    def _input(name, dims):
        i = types.SimpleNamespace(name=name); i.type = types.SimpleNamespace(tensor_type=types.SimpleNamespace(shape=types.SimpleNamespace(dim=dims)))
        return i

    protos = {
        "det.onnx": types.SimpleNamespace(graph=types.SimpleNamespace(input=[_input("input.1", [_Dim(p="batch"), _Dim(3), _Dim(p="h"), _Dim(p="w")])])),
        "rec.onnx": types.SimpleNamespace(graph=types.SimpleNamespace(input=[_input("input.1", [_Dim(1), _Dim(3), _Dim(112), _Dim(112)])])),
    }
    fixed_calls = []
    fake_onnx = types.SimpleNamespace(load=lambda path: protos[path.split("/")[-1]], save=lambda proto, path: open(path, "wb").write(b"fixed"))
    fake_utils = types.SimpleNamespace(make_input_shape_fixed=lambda g, n, d: fixed_calls.append((n, d)), fix_output_shapes=lambda p: None)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxruntime.tools", types.SimpleNamespace(onnx_model_utils=fake_utils))
    monkeypatch.setitem(sys.modules, "onnxruntime.tools.onnx_model_utils", fake_utils)
    for n in protos:
        (tmp_path / n).write_bytes(b"m")
    out = fixed_shape_model(str(tmp_path / "det.onnx"), "*[1,3,640,640]", "QNNExecutionProvider")
    assert out.endswith("det.onnx.QNNExecutionProvider.fixed.onnx") and fixed_calls == [("input.1", [1, 3, 640, 640])]
    assert fixed_shape_model(str(tmp_path / "rec.onnx"), "*[1,3,640,640]", "QNNExecutionProvider") == str(tmp_path / "rec.onnx")
    assert fixed_shape_model(str(tmp_path / "rec.onnx"), "images[1,3,448,448]", "QNNExecutionProvider") == str(tmp_path / "rec.onnx")

# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Runtime execution providers, chosen once for every adapter.

OpenNVR Accelerate ships each ONNX adapter in hardware *flavours* (the same
code, a different ``onnxruntime`` wheel and vendor user-space: ``cpu``,
``intel`` → OpenVINO on the iGPU/NPU, ``nvidia`` → CUDA). This module is the
one place that turns "which providers, on which device" into what
``onnxruntime.InferenceSession(providers=...)`` takes, so no adapter carries a
hardcoded ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` again.

The setting is ``ADAPTER_ONNX_PROVIDERS``: a comma-separated provider list,
each with an optional ``:device`` that becomes the provider's device option::

    OpenVINOExecutionProvider:NPU                 -> {"device_type": "NPU"}
    OpenVINOExecutionProvider:AUTO:NPU,GPU,CPU    -> {"device_type": "AUTO:NPU,GPU,CPU"}
    CUDAExecutionProvider:1                       -> {"device_id": "1"}

A token that is not a provider name continues the previous device, so the
OpenVINO multi-device forms survive the comma split. CPU is always appended
as a fallback. Unset means "whatever this build has, in ORT's own order",
which on the cpu flavour is CPU and on the intel flavour would be OpenVINO's
*CPU device* — so the intel overlay always sets the device explicitly.

ORT drops a provider it cannot initialise (no device, no driver) and carries
on *silently*. :func:`effective_providers` compares what was asked with what
the session runs on, logs one warning, and feeds the hardware-evaluation
payload — the "accelerated on paper, running on CPU" state is never hidden.

Mirrors ``detect_pipeline/onnx_detector.py`` in open-nvr; keep the two in
step.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("opennvr_adapter_sdk.providers")

ENV_VAR = "ADAPTER_ONNX_PROVIDERS"
#: Set to ``1`` to let OpenVINO accelerator devices compile dynamic-shape
#: graphs as they are. Off by default: the NPU plugin cannot run dynamic
#: shapes at all, and an Intel iGPU driver has been seen to hard-reset the
#: host on a dynamic compile — static compiles are the safe default.
DYNAMIC_SHAPES_ENV = "ADAPTER_ONNX_DYNAMIC_SHAPES"
CPU_EP = "CPUExecutionProvider"
OPENVINO_EP = "OpenVINOExecutionProvider"
# Providers that are *not* an accelerator even though they are not CPU.
_NOT_ACCELERATORS = {CPU_EP, "AzureExecutionProvider"}
# The device option each provider takes; others refuse the shorthand.
DEVICE_OPTION = {
    "OpenVINOExecutionProvider": "device_type",
    "CUDAExecutionProvider": "device_id",
    "TensorrtExecutionProvider": "device_id",
    "ROCMExecutionProvider": "device_id",
    "MIGraphXExecutionProvider": "device_id",
    # Qualcomm AI Runtime: ``QNNExecutionProvider:htp`` is the Hexagon NPU,
    # ``:gpu`` the Adreno, ``:cpu`` QNN's reference backend (debugging only).
    "QNNExecutionProvider": "backend_type",
}
QNN_EP = "QNNExecutionProvider"

# Plugin execution providers (ONNX Runtime >= 1.23): a separately installed
# wheel the runtime must REGISTER before a session can use it, and that
# ``get_available_providers()`` never lists. Provider name -> the package
# that ships the library. The ``qualcomm`` flavour installs onnxruntime-qnn
# next to plain onnxruntime (it is a plugin, not a replacement wheel).
PLUGIN_EPS = {QNN_EP: "onnxruntime_qnn"}
_registered_plugin_eps: set[str] = set()

#: Extra provider options, ``key=value,key=value``, on top of the flavour's
#: defaults (QNN: htp_performance_mode, vtcm_mb, enable_htp_fp16_precision …).
PROVIDER_OPTIONS_ENV = "ADAPTER_ONNX_PROVIDER_OPTIONS"
# What a plugin provider gets under the ``:device`` shorthand. QNN's HTP:
# sustained clocks (burst throttles under steady load), fp16 for an fp32
# model (the HTP has no fp32 path; without it every node falls back to
# CPU), the slowest graph finalisation (once per session, cached — see
# ``plugin_context_path``). Measured on a QCS8275: yolo26n 8.6 ms.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    QNN_EP: {
        "backend_type": "htp",
        "htp_performance_mode": "sustained_high_performance",
        "enable_htp_fp16_precision": "1",
        "htp_graph_finalization_optimization_mode": "3",
    },
}

#: Private option static_shapes() leaves on a QNN entry: ``name[1,3,448,448],…``.
#: plugin_session() consumes it (never handed to the runtime).
FIXED_SHAPES_KEY = "_opennvr_fixed_shapes"

ProviderEntry = Any  # str | tuple[str, dict[str, str]]


def parse_provider_specs(requested) -> list[tuple[str, str | None]]:
    """``"A:dev,B"`` or ``["A:dev", "B"]`` → ``[("A", "dev"), ("B", None)]``.

    A token that does not end in ``ExecutionProvider`` continues the previous
    token's device (``OpenVINOExecutionProvider:AUTO:NPU,GPU`` → one spec).
    """
    if requested is None:
        return []
    if isinstance(requested, str):
        tokens = [t.strip() for t in requested.split(",") if t.strip()]
    else:
        tokens = []
        for item in requested:
            if isinstance(item, tuple):
                name, opts = item
                dev = None
                if isinstance(opts, dict):
                    dev = opts.get(DEVICE_OPTION.get(name, ""))
                tokens.append(f"{name}:{dev}" if dev else name)
            else:
                tokens.append(str(item).strip())
    specs: list[tuple[str, str | None]] = []
    for tok in tokens:
        if not tok:
            continue
        name, _, device = tok.partition(":")
        if name.endswith("ExecutionProvider") or not specs:
            specs.append((name, device or None))
        else:
            prev_name, prev_dev = specs[-1]
            specs[-1] = (prev_name, f"{prev_dev},{tok}" if prev_dev else tok)
    return specs


def provider_entry(name: str, device: str | None) -> ProviderEntry:
    """One spec as ORT takes it: the bare name, or ``(name, {option: device})``."""
    if not device:
        return name
    option = DEVICE_OPTION.get(name)
    if option is None:
        log.warning("ORT provider %s takes no device; ignoring %r", name, device)
        return name
    options = {option: device}
    if name == OPENVINO_EP and device.upper() != "CPU" and os.getenv(DYNAMIC_SHAPES_ENV) != "1":
        # Compile the graph for the shapes of the first inference instead
        # of as a dynamic graph (recompiled if a later shape differs).
        options["disable_dynamic_shapes"] = "True"
    return (name, options)


def provider_name(entry: ProviderEntry) -> str:
    return entry[0] if isinstance(entry, tuple) else str(entry)


def resolve_providers(requested, available) -> list[ProviderEntry]:
    """``requested ∩ available`` in the requested order, CPU always last.

    ``requested`` may be a comma string, a list, or ``None`` ("let ORT pick
    from what it has"). A requested provider this build does not have is
    skipped with a warning rather than failing the adapter.
    """
    available = list(available) if available else []
    specs = parse_provider_specs(requested)
    if not specs:
        provs: list[ProviderEntry] = list(available) if available else [CPU_EP]
    else:
        provs = []
        for name, device in specs:
            if available and name not in available:
                log.warning("ORT provider %s not available in this build; skipping", name)
                continue
            provs.append(provider_entry(name, device))
    if CPU_EP not in [provider_name(p) for p in provs]:
        provs.append(CPU_EP)
    return provs


def wants_static_shapes(entries) -> bool:
    """True when the first provider is an accelerator that compiles the graph
    for fixed shapes — an OpenVINO device other than ``CPU``, or the QNN HTP
    — so a dynamic-shape graph should be pinned with :func:`static_shapes`
    before the session is built."""
    for entry in entries or []:
        name = provider_name(entry)
        opts = entry[1] if isinstance(entry, tuple) and isinstance(entry[1], dict) else {}
        if name == OPENVINO_EP:
            device = opts.get("device_type", "")
            return bool(device) and device.upper() != "CPU"
        if name == QNN_EP:
            return opts.get("backend_type", "htp").lower() != "cpu"
        return False
    return False


def static_shapes(entries, shapes) -> list[ProviderEntry]:
    """Pin a dynamic-shape graph for OpenVINO: ``shapes`` maps each input name
    to its full shape, e.g. ``{"images": [1, 3, 448, 448]}``.

    Returns a new provider list in which every OpenVINO entry carries
    ``reshape_input`` (the plugin reshapes the graph once, before compiling)
    and non-OpenVINO entries are untouched. The session then accepts ONLY
    those shapes — the adapter must size its frames to match. Needed for
    the NPU, which has no dynamic-shape support, and the safe choice on an
    iGPU.
    """
    spec = ",".join(
        f"{name}[{','.join(str(int(d)) for d in dims)}]" for name, dims in dict(shapes).items()
    )
    out: list[ProviderEntry] = []
    for entry in entries or []:
        name = provider_name(entry)
        if name == QNN_EP:
            # QNN has no reshape option: plugin_session writes a fixed-shape
            # copy of the model before compiling (FIXED_SHAPES_KEY).
            options = dict(entry[1]) if isinstance(entry, tuple) else {}
            options[FIXED_SHAPES_KEY] = spec
            out.append((QNN_EP, options))
            continue
        if name != OPENVINO_EP:
            out.append(entry)
            continue
        options = dict(entry[1]) if isinstance(entry, tuple) else {}
        options.pop("disable_dynamic_shapes", None)
        options["reshape_input"] = spec
        out.append((OPENVINO_EP, options))
    return out


def describe_providers(entries) -> str:
    """``OpenVINOExecutionProvider(NPU), CPUExecutionProvider`` — for logs and the UI."""
    parts = []
    for e in entries or []:
        if isinstance(e, tuple):
            name, opts = e
            dev = next(iter((opts or {}).values()), None)
            parts.append(f"{name}({dev})" if dev else name)
        else:
            parts.append(str(e))
    return ", ".join(parts)


def requested_providers(default: str | None = None) -> str | None:
    """The operator's setting, from ``ADAPTER_ONNX_PROVIDERS`` (else ``default``)."""
    value = os.environ.get(ENV_VAR, "").strip()
    return value or default


def available_providers(requested=None) -> list[str]:
    """This build's providers — plus the plugin providers ``requested`` names,
    registered on the way (``get_available_providers()`` never lists those)."""
    try:
        import onnxruntime as ort  # lazy: optional dependency

        avail = list(ort.get_available_providers())
    except Exception:
        return []
    for name in register_plugin_providers(requested) | _registered_plugin_eps:
        if name not in avail:
            avail.append(name)
    return avail


def register_plugin_providers(requested) -> set[str]:
    """Register the plugin EP libraries ``requested`` names (see ``PLUGIN_EPS``)
    and return the names now usable, the already-registered ones included.

    Process-wide and idempotent. A plugin whose package is not in this
    build (the cpu flavour asked for QNN) is reported once and skipped, so
    the request degrades to the next provider like a built-in one the
    wheel lacks. Registering also installs :func:`install_plugin_shim`, so
    a library that builds its own sessions from a ``providers=`` list
    (InsightFace, fast-plate-ocr, onnx-asr) gets the plugin too.
    """
    usable: set[str] = set()
    for name, _device in parse_provider_specs(requested):
        if name not in PLUGIN_EPS:
            continue
        if name in _registered_plugin_eps:
            usable.add(name)
            continue
        pkg = PLUGIN_EPS[name]
        try:
            import importlib

            import onnxruntime as ort

            mod = importlib.import_module(pkg)
            ort.register_execution_provider_library(name, mod.get_library_path())
        except ImportError:
            log.warning("ORT provider %s needs the %s package (the matching image flavour); "
                        "not installed — skipping", name, pkg)
            continue
        except Exception as exc:  # the runtime refused the library
            log.warning("ORT could not register plugin provider %s: %s", name, exc)
            continue
        _registered_plugin_eps.add(name)
        usable.add(name)
        log.info("ORT plugin provider %s registered from %s", name, pkg)
    if usable:
        install_plugin_shim()
    return usable


def provider_options(entry: ProviderEntry, env=None) -> dict[str, str]:
    """The options a provider entry carries: the provider's defaults underneath,
    the entry's own on top, ``ADAPTER_ONNX_PROVIDER_OPTIONS`` (``k=v,k=v``)
    last. For QNN the ``backend_type`` shorthand becomes the package's
    ``backend_path``: the documented ``backend_type`` failed device creation
    on a QCS8275 (``QNN_DEVICE_ERROR_INVALID_CONFIG``) where the path worked."""
    name = provider_name(entry)
    opts = dict(PROVIDER_DEFAULTS.get(name, {}))
    if isinstance(entry, tuple) and isinstance(entry[1], dict):
        opts.update({k: str(v) for k, v in entry[1].items()})
    raw = (env if env is not None else os.environ).get(PROVIDER_OPTIONS_ENV, "")
    for item in raw.split(","):
        k, sep, v = item.strip().partition("=")
        if sep and k:
            opts[k] = v
    if name == QNN_EP and "backend_path" not in opts:
        path = _qnn_backend_path(opts.get("backend_type", "htp"))
        if path:
            opts["backend_path"] = path
            opts.pop("backend_type", None)
    return opts


def _qnn_backend_path(backend: str) -> str | None:
    try:
        import onnxruntime_qnn as q
    except ImportError:
        return None
    getter = getattr(q, f"get_qnn_{backend.strip().lower()}_path", None)
    return getter() if getter else None


def plugin_context_path(model, provider: str) -> str:
    """Where a plugin provider's compiled graph for ``model`` is cached: in
    :func:`accel_cache_dir` (a hidden directory inside the model's, else the
    temp dir); the name carries the provider so flavours never share one. QNN compiles
    the graph at session build (seconds); the EP-context dump loads in a
    fraction of that for every later session. A model given as bytes is
    keyed by its hash."""
    import hashlib
    import tempfile

    if isinstance(model, (bytes, bytearray)):
        return os.path.join(accel_cache_dir(None),
                            f"{hashlib.sha256(model).hexdigest()[:16]}.{provider}.ctx.onnx")
    path = str(model)
    return os.path.join(accel_cache_dir(path), f"{os.path.basename(path)}.{provider}.ctx.onnx")


#: Where the SDK keeps a provider's derived files (compiled graphs, pinned
#: copies). Default: ``.opennvr-accel/`` INSIDE the model's directory — never
#: beside the model, because libraries glob ``*.onnx`` there (InsightFace
#: loads every file in its pack directory as a model; measured); a model
#: given as bytes, or a read-only directory, goes under the temp dir.
ACCEL_CACHE_ENV = "ADAPTER_ACCEL_CACHE_DIR"


def accel_cache_dir(model_path: str | None) -> str:
    import tempfile

    d = os.environ.get(ACCEL_CACHE_ENV, "").strip()
    if not d:
        base = os.path.dirname(os.path.abspath(model_path)) if model_path else tempfile.gettempdir()
        if os.path.basename(base) == ".opennvr-accel":
            d = base  # a derived file's own derivatives (the pinned copy's compiled graph) stay flat
        elif model_path and os.access(base, os.W_OK):
            d = os.path.join(base, ".opennvr-accel")
        else:
            d = os.path.join(tempfile.gettempdir(), "opennvr-accel")
    os.makedirs(d, exist_ok=True)
    return d


def fixed_shape_model(model, spec: str, provider: str, *, adapter: str = "adapter"):
    """A copy of ``model`` with the inputs in ``spec`` (``name[1,3,448,448],…``;
    ``*`` is the first input: its symbolic dims take the spec's values, its
    fixed dims stay) made static, written next to the model (or under the temp dir) as
    ``<model>.<provider>.fixed.onnx`` and reused while the model is older
    than it. QNN compiles for fixed shapes only; a dynamic-axes export
    otherwise falls back to CPU node by node. Needs the ``onnx`` package
    (in the qualcomm flavour); without it, or for a model given as bytes,
    the model is returned as is with a warning."""
    if not isinstance(model, str):
        log.warning("%s: cannot pin shapes of an in-memory model for %s; compiling the dynamic graph", adapter, provider)
        return model
    base = os.path.join(accel_cache_dir(model), f"{os.path.basename(model)}.{provider}.fixed.onnx")
    try:
        if os.path.exists(base) and os.path.getmtime(base) >= os.path.getmtime(model):
            return base
        import onnx  # noqa: F401 — the tool below needs it
        from onnxruntime.tools import onnx_model_utils as u

        proto = onnx.load(model)
        inputs = {i.name: i for i in proto.graph.input}

        def _dynamic(inp) -> bool:
            dims = inp.type.tensor_type.shape.dim
            return any(d.dim_param or d.dim_value <= 0 for d in dims)

        pinned = 0
        for item in spec.split("]"):
            item = item.strip(", ")
            if not item:
                continue
            name, _, dims = item.partition("[")
            values = [int(d) for d in dims.split(",")]
            if name == "*":
                # "the first input's SYMBOLIC dims take these values; its fixed
                # dims stay" — for a library that builds several sessions from
                # one provider list (InsightFace: a detector with dynamic
                # height/width, a recogniser and landmark nets with a dynamic
                # batch and fixed 112/192/96 sizes) the one spec pins each
                # graph to the shape it really takes. A fixed input is left alone.
                inp = proto.graph.input[0]
                name = inp.name
                model_dims = inp.type.tensor_type.shape.dim
                if not _dynamic(inp) or len(model_dims) != len(values):
                    continue
                values = [
                    (v if (d.dim_param or d.dim_value <= 0) else d.dim_value)
                    for d, v in zip(model_dims, values)
                ]
            if name not in inputs:
                continue  # another model built from the same list
            u.make_input_shape_fixed(proto.graph, name, values)
            pinned += 1
        if not pinned:
            return model
        u.fix_output_shapes(proto)
        tmp = f"{base}.{os.getpid()}.tmp"
        onnx.save(proto, tmp)
        os.replace(tmp, base)
        log.info("%s: pinned %s to %s for %s -> %s", adapter, os.path.basename(model), spec, provider, base)
        return base
    except ImportError:
        log.warning("%s: the onnx package is not installed; cannot pin %s for %s, compiling the dynamic graph",
                    adapter, os.path.basename(model), provider)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: could not pin %s to %s (%s); compiling the dynamic graph", adapter, os.path.basename(model), spec, exc)
    return model


def _plugin_build_plan(ort, model, entries, sess_options=None, *, adapter: str = "adapter"):
    """Attach the plugin provider(s) in ``entries`` to the session options
    BY DEVICE and decide what to load: ``(model_or_ctx, opts, classic, tmp_dump, ctx, provider)``.

    ``add_provider_for_devices`` is the only way in: a plugin named in
    ``providers=`` is ignored and ``add_provider`` by name is refused. ``ctx``
    is the compiled-graph cache; when it exists it is what gets loaded, else
    the runtime is asked to dump it at ``tmp_dump`` (renamed into place by
    :func:`_plugin_build_finish`). ``provider`` is None when nothing attached.
    """
    import threading

    opts = sess_options if sess_options is not None else ort.SessionOptions()
    plugins = [e for e in entries or [] if provider_name(e) in _registered_plugin_eps]
    classic = [e for e in entries or [] if provider_name(e) not in PLUGIN_EPS]
    attached = []
    for e in plugins:
        name = provider_name(e)
        devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
        if not devices:
            log.warning("%s: ORT plugin provider %s reports no device on this host; skipping", adapter, name)
            continue
        popts = provider_options(e)
        spec = popts.pop(FIXED_SHAPES_KEY, None)
        if spec:
            model = fixed_shape_model(model, spec, name, adapter=adapter)
        opts.add_provider_for_devices(devices, popts)
        attached.append(name)
    if not attached:
        return model, opts, classic, None, None, None
    provider = attached[0]
    ctx = plugin_context_path(model, provider)
    if os.path.exists(ctx):
        return ctx, opts, classic, None, ctx, provider
    tmp = f"{ctx}.{os.getpid()}.{threading.get_ident()}.tmp"
    opts.add_session_config_entry("ep.context_enable", "1")
    opts.add_session_config_entry("ep.context_file_path", tmp)
    opts.add_session_config_entry("ep.context_embed_mode", "1")
    return model, opts, classic, tmp, ctx, provider


def _plugin_build_finish(sess, tmp, ctx, provider, *, adapter: str = "adapter") -> None:
    """After the session is built: keep the compiled-graph dump and mark the
    session. ``_opennvr_plugin_compiled`` is True when the provider took the
    graph, False when it compiled nothing — the only reliable sign, since
    ``get_providers()`` keeps naming a plugin whose device failed (measured:
    QNN refused, every node on CPU, the session still said QNN)."""
    if provider is None:
        sess._opennvr_plugin_compiled = None
        return
    if tmp is None:  # loaded the cache
        log.info("%s: %s loaded the compiled graph %s", adapter, provider, ctx)
        sess._opennvr_plugin_compiled = True
        return
    if os.path.exists(tmp):
        try:
            os.replace(tmp, ctx)
            log.info("%s: %s compiled the graph, cached at %s", adapter, provider, ctx)
        except OSError as exc:
            log.warning("%s: could not keep the compiled graph at %s: %s", adapter, ctx, exc)
        sess._opennvr_plugin_compiled = True
    else:
        sess._opennvr_plugin_compiled = False


def plugin_session(model, entries, sess_options=None, *, adapter: str = "adapter", **kw):
    """Build an ``InferenceSession`` with the plugin provider(s) in ``entries``
    attached by device, through the compiled-graph cache (see
    :func:`_plugin_build_plan`). A cache the runtime rejects (another QNN
    version, another SoC) is deleted and the model compiled again."""
    import onnxruntime as ort

    original = _ORIGINAL_SESSION or ort.InferenceSession
    for attempt in (1, 2):
        load, opts, classic, tmp, ctx, provider = _plugin_build_plan(
            ort, model, entries, sess_options, adapter=adapter)
        try:
            sess = original(load, sess_options=opts, providers=classic or None, **kw) if provider is None \
                else original(load, sess_options=opts, **kw)
        except Exception as exc:
            if provider is not None and tmp is None and attempt == 1:
                log.warning("%s: compiled graph %s rejected (%s); rebuilding from the model", adapter, ctx, exc)
                try:
                    os.remove(ctx)
                except OSError:
                    pass
                sess_options = None  # the options carry the stale attach; start over
                continue
            raise
        _plugin_build_finish(sess, tmp, ctx, provider, adapter=adapter)
        return sess
    raise RuntimeError("unreachable")


_ORIGINAL_SESSION = None


def install_plugin_shim() -> None:
    """Make ``onnxruntime.InferenceSession`` a SUBCLASS that routes a
    ``providers=`` list naming a registered plugin provider through the plugin
    API (see :func:`_plugin_build_plan`); any other list goes straight through.

    The adapters' libraries (InsightFace, fast-plate-ocr, onnx-asr …) build
    their own sessions from the provider list they are handed, and ORT
    silently ignores a plugin named there (measured) — so the only way to
    give them the NPU is to meet them at the constructor. A subclass rather
    than a function because libraries also subclass it (InsightFace's
    PickableInferenceSession) and pickle it. Idempotent."""
    global _ORIGINAL_SESSION
    if _ORIGINAL_SESSION is not None:
        return
    import onnxruntime as ort

    original = ort.InferenceSession
    _ORIGINAL_SESSION = original

    class InferenceSession(original):  # noqa: N801 — stands in for the runtime's class
        def __init__(self, model, sess_options=None, providers=None, provider_options=None, **kw):
            if providers and provider_options:
                entries = [(p, dict(o)) if isinstance(p, str) and o else p
                           for p, o in zip(providers, list(provider_options) + [None] * len(providers))]
            else:
                entries = list(providers or [])
            if not any(provider_name(e) in _registered_plugin_eps for e in entries):
                super().__init__(model, sess_options=sess_options, providers=providers,
                                 provider_options=provider_options, **kw)
                return
            for attempt in (1, 2):
                load, opts, classic, tmp, ctx, provider = _plugin_build_plan(ort, model, entries, sess_options)
                try:
                    if provider is None:
                        super().__init__(load, sess_options=opts, providers=classic or None, **kw)
                    else:
                        super().__init__(load, sess_options=opts, **kw)
                except Exception as exc:
                    if provider is not None and tmp is None and attempt == 1:
                        log.warning("compiled graph %s rejected (%s); rebuilding from the model", ctx, exc)
                        try:
                            os.remove(ctx)
                        except OSError:
                            pass
                        sess_options = None
                        continue
                    raise
                _plugin_build_finish(self, tmp, ctx, provider)
                return

    InferenceSession.__module__ = original.__module__
    InferenceSession.__qualname__ = original.__qualname__
    InferenceSession.__wrapped__ = original
    ort.InferenceSession = InferenceSession
    log.info("ORT plugin shim installed: providers lists naming %s go through the plugin API",
             ", ".join(sorted(_registered_plugin_eps)))


def accelerator_available(available=None) -> bool:
    """Does this *build* carry an accelerator provider at all (CUDA, OpenVINO,
    TensorRT, ROCm, a registered plugin …)? The §8 ``gpu`` permission declares
    exactly this."""
    provs = list(available) if available is not None else available_providers()
    return any(p not in _NOT_ACCELERATORS for p in provs)


def session_providers(requested=None) -> list[ProviderEntry]:
    """What to pass as ``providers=`` for an ``InferenceSession``: the env
    setting (or ``requested``) resolved against this build's providers. A
    plugin provider in the request is registered here (and the constructor
    shim installed), so the list works in a library's ``providers=`` too."""
    req = requested if requested is not None else requested_providers()
    register_plugin_providers(req)
    return resolve_providers(req, available_providers())


def build_with_fallback(build, requested, *, adapter: str = "adapter"):
    """Run ``build(providers)`` with the requested providers; if that raises
    and an accelerator was requested, log the error and build again on CPU.

    Returns ``(obj, providers_used, fallback_error)``: ``fallback_error`` is
    ``None`` when the first attempt succeeded, else the error text (hand it
    to :func:`effective_providers`). An accelerator that cannot even start
    — device missing from the container, driver absent, plugin crash —
    must never take the adapter down with it: the adapter serves on CPU
    and ``/hardware/evaluation`` says why. A failure with no accelerator
    requested is a real error and propagates.
    """
    req = list(requested) if requested else [CPU_EP]
    try:
        return build(req), req, None
    except Exception as exc:  # noqa: BLE001 — any failure of the accelerated path
        wanted = provider_name(req[0]) if req else CPU_EP
        if wanted in _NOT_ACCELERATORS:
            raise
        error = f"{type(exc).__name__}: {exc}"[:400]
        log.error(
            "%s: could not start on %s — %s. Falling back to CPU; the adapter works "
            "but is not accelerated. Check the device is passed through to the "
            "container and the image flavour matches the host.",
            adapter, describe_providers(req[:1]), error,
        )
        return build([CPU_EP]), [CPU_EP], error


def effective_providers(session, requested=None, *, adapter: str = "adapter",
                        fallback_error: str | None = None) -> dict[str, Any]:
    """Compare the request with what ``session`` really runs on.

    Returns the hardware-evaluation payload fragment::

        {"onnxruntime_providers": [...available...],
         "requested_providers": "OpenVINOExecutionProvider(NPU), CPUExecutionProvider",
         "effective_providers": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
         "accelerator_in_use": True,
         "accelerator_requested": True}

    and logs ONE warning when the first requested provider is not the one the
    session runs on — ORT's silent fallback, made loud.
    """
    try:
        effective = list(session.get_providers())
    except Exception:
        effective = []
    if getattr(session, "_opennvr_plugin_compiled", None) is False:
        # The plugin's device never came up and every node is on CPU, yet
        # the session still lists it (see plugin_session).
        effective = [p for p in effective if p not in PLUGIN_EPS]
    req = list(requested) if requested else []
    wanted = provider_name(req[0]) if req else None
    in_use = bool(effective) and effective[0] not in _NOT_ACCELERATORS
    if wanted and wanted not in _NOT_ACCELERATORS and (not effective or effective[0] != wanted):
        log.warning(
            "%s: ONNX Runtime could not use %s — running on %s. Check the accelerator's "
            "user-space driver is in the image (the matching image flavour) and its "
            "device is passed through to the container.",
            adapter, describe_providers(req[:1]), ", ".join(effective) or "nothing",
        )
    elif wanted:
        log.info("%s: ONNX Runtime running on %s", adapter, describe_providers(req[:1]))
    report = {
        "onnxruntime_providers": available_providers(),
        "requested_providers": describe_providers(req),
        "effective_providers": effective,
        "accelerator_in_use": in_use,
        "accelerator_requested": bool(wanted) and wanted not in _NOT_ACCELERATORS,
    }
    if fallback_error:
        # build_with_fallback() rebuilt on CPU: the request is what the
        # operator asked for, the session is CPU, and here is why.
        report["accelerator_requested"] = True
        report["accelerator_fallback_error"] = fallback_error
    return report


__all__ = [
    "ENV_VAR", "CPU_EP", "DEVICE_OPTION", "QNN_EP", "PLUGIN_EPS", "PROVIDER_OPTIONS_ENV",
    "PROVIDER_DEFAULTS", "register_plugin_providers", "provider_options", "plugin_context_path",
    "plugin_session", "install_plugin_shim", "fixed_shape_model", "FIXED_SHAPES_KEY",
    "accel_cache_dir", "ACCEL_CACHE_ENV",
    "parse_provider_specs", "provider_entry", "provider_name", "resolve_providers",
    "describe_providers", "requested_providers", "available_providers",
    "accelerator_available", "session_providers", "effective_providers",
    "build_with_fallback", "static_shapes", "wants_static_shapes",
]

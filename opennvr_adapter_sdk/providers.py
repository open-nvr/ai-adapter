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
}

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
    """True when the first provider is an OpenVINO accelerator device (anything
    but ``CPU``): a dynamic-shape graph should be pinned with
    :func:`static_shapes` before the session is built."""
    for entry in entries or []:
        if provider_name(entry) != OPENVINO_EP:
            return False
        device = entry[1].get("device_type", "") if isinstance(entry, tuple) else ""
        return bool(device) and device.upper() != "CPU"
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
        if provider_name(entry) != OPENVINO_EP:
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


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort  # lazy: optional dependency

        return list(ort.get_available_providers())
    except Exception:
        return []


def accelerator_available(available=None) -> bool:
    """Does this *build* carry an accelerator provider at all (CUDA, OpenVINO,
    TensorRT, ROCm …)? The §8 ``gpu`` permission declares exactly this."""
    provs = list(available) if available is not None else available_providers()
    return any(p not in _NOT_ACCELERATORS for p in provs)


def session_providers(requested=None) -> list[ProviderEntry]:
    """What to pass as ``providers=`` for an ``InferenceSession``: the env
    setting (or ``requested``) resolved against this build's providers."""
    return resolve_providers(
        requested if requested is not None else requested_providers(), available_providers()
    )


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
    "ENV_VAR", "CPU_EP", "DEVICE_OPTION",
    "parse_provider_specs", "provider_entry", "provider_name", "resolve_providers",
    "describe_providers", "requested_providers", "available_providers",
    "accelerator_available", "session_providers", "effective_providers",
    "build_with_fallback", "static_shapes", "wants_static_shapes",
]

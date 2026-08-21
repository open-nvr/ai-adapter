# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Prometheus metrics (SDK consolidated).

Replaces the per-adapter ``metrics.py`` files. Latency buckets are
configurable per-adapter — the default covers TTS / vision detection
/ ASR in one shot (10ms - 60s). Adapters with different latency
profiles can pass custom buckets to ``Metrics(...)``.
"""
from __future__ import annotations

import threading
from collections import defaultdict


# Default bucket set — spans 10ms (cached short clip) → 60s (long
# audio on CPU). YOLOv8 on GPU lives in the 5-50ms range; Whisper
# CPU lives in 1-30s. Both fit.
DEFAULT_LATENCY_BUCKETS_SECONDS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)


class Metrics:
    """In-process metrics registry. Thread-safe for the counter and
    gauge operations the adapter performs concurrently.

    All adapters share this implementation; latency buckets are the
    only tunable. Prometheus exposition format ("text/plain version
    0.0.4") is emitted by ``render()`` — no ``prometheus_client``
    dependency required.
    """

    KNOWN_OUTCOMES: tuple[str, ...] = (
        "ok",
        "model_error",
        "provider_error",
        "transport_error",
        "refused",
    )

    def __init__(
        self,
        *,
        latency_buckets_seconds: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_SECONDS,
        known_tasks: tuple[str, ...] | list[str] = (),
    ) -> None:
        self._latency_buckets = latency_buckets_seconds
        self._lock = threading.Lock()
        # ``task`` label — the second axis of every infer series. The label
        # set is CLOSED: only tasks the adapter advertises (plus "" for
        # unattributable calls and "other" for anything else) ever become a
        # series. Task strings arrive in request payloads, i.e. they are
        # client-controlled — an open label set would let any client mint
        # unbounded series (a cardinality bomb) by sending random task names.
        self._known_tasks = frozenset(str(t) for t in known_tasks)
        # (outcome, task) → count
        self._infer_total: dict[tuple[str, str], int] = defaultdict(int)
        # task → {"buckets": {ub: n}, "inf": n, "sum": s, "count": n}
        self._lat: dict[str, dict] = {}
        # Model identity labels (adapter_model_info) — set at lifespan
        # startup and refreshed on every /capabilities build, so a live
        # fingerprint drift (§11.3) is visible on /metrics too.
        self._model_labels: dict[str, str] = {}
        self._model_loaded: int = 0
        self._stream_active: int = 0
        self._inflight: int = 0
        self._queue_depth: int = 0
        # Adapter-defined series (see register_counter / register_histogram).
        self._custom_counters: dict[str, dict] = {}
        self._custom_hists: dict[str, dict] = {}

        # Pre-seed all outcomes at 0 (task="") so Prometheus sees them
        # even before the first request.
        for outcome in self.KNOWN_OUTCOMES:
            self._infer_total[(outcome, "")] = 0
        self._lat[""] = self._new_hist()

    def _new_hist(self) -> dict:
        return {
            "buckets": {ub: 0 for ub in self._latency_buckets},
            "inf": 0, "sum": 0.0, "count": 0,
        }

    def _norm_task(self, task: str) -> str:
        task = str(task or "")
        if not task or task in self._known_tasks:
            return task
        return "other"

    def record_infer(
        self, outcome: str, latency_seconds: float, task: str = "",
    ) -> None:
        if outcome not in self.KNOWN_OUTCOMES:
            outcome = "model_error"
        task = self._norm_task(task)
        with self._lock:
            self._infer_total[(outcome, task)] += 1
            hist = self._lat.get(task)
            if hist is None:
                hist = self._lat[task] = self._new_hist()
            for ub in self._latency_buckets:
                if latency_seconds <= ub:
                    hist["buckets"][ub] += 1
            hist["inf"] += 1
            hist["sum"] += latency_seconds
            hist["count"] += 1

    # ── Adapter-defined (domain) metrics ───────────────────────────
    #
    # Each model family has its own "is it healthy AND efficient" numbers —
    # detections per class for a detector, audio realtime-factor for
    # STT/TTS, generated volume for a captioner, upstream latency for a
    # proxy. Adapters register those series ONCE (typically in load()) and
    # then increment/observe them; the SDK renders them next to the
    # standard set so one scrape answers everything. Same discipline as the
    # ``task`` label: names are fixed at registration, and any counter
    # label is a CLOSED value set ("other" catches the rest) — nothing a
    # client sends can mint unbounded series.

    def register_counter(
        self, name: str, help_text: str, *,
        label_key: str | None = None,
        allowed_values: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Register a custom counter. ``name`` must start with ``adapter_``.
        With ``label_key``, series are split by that one label whose values
        are limited to ``allowed_values`` (+ automatic "other")."""
        if not name.startswith("adapter_"):
            raise ValueError(f"custom metric {name!r} must start with 'adapter_'")
        with self._lock:
            if name in self._custom_counters:
                return                      # idempotent: load() retries must not reset counts
            self._custom_counters[name] = {
                "help": help_text,
                "label_key": label_key,
                "allowed": frozenset(str(v) for v in allowed_values),
                "values": defaultdict(float),   # label value ("" if none) → count
            }

    def inc_counter(
        self, name: str, value: float = 1.0, label_value: str | None = None,
    ) -> None:
        with self._lock:
            reg = self._custom_counters.get(name)
            if reg is None:
                raise ValueError(f"custom counter {name!r} was never registered")
            key = ""
            if reg["label_key"] is not None:
                key = str(label_value or "")
                if key and key not in reg["allowed"]:
                    key = "other"
            reg["values"][key] += value

    def register_histogram(
        self, name: str, help_text: str, *,
        buckets: tuple[float, ...] | list[float],
    ) -> None:
        """Register a custom (label-free) histogram with explicit buckets."""
        if not name.startswith("adapter_"):
            raise ValueError(f"custom metric {name!r} must start with 'adapter_'")
        ubs = tuple(sorted(float(b) for b in buckets))
        if not ubs:
            raise ValueError(f"custom histogram {name!r} needs at least one bucket")
        with self._lock:
            if name in self._custom_hists:
                return                      # idempotent: load() retries must not reset counts
            self._custom_hists[name] = {
                "help": help_text,
                "buckets": {ub: 0 for ub in ubs},
                "ubs": ubs, "inf": 0, "sum": 0.0, "count": 0,
            }

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            hist = self._custom_hists.get(name)
            if hist is None:
                raise ValueError(f"custom histogram {name!r} was never registered")
            for ub in hist["ubs"]:
                if value <= ub:
                    hist["buckets"][ub] += 1
            hist["inf"] += 1
            hist["sum"] += value
            hist["count"] += 1

    def set_model_info(self, **labels: str | None) -> None:
        """Set the ``adapter_model_info`` identity labels. Empty/None values
        are dropped; the metric's value is always 1 — the labels ARE the
        payload (Prometheus info-metric convention, like ``build_info``)."""
        cleaned = {
            str(k): str(v) for k, v in labels.items() if v is not None and str(v) != ""
        }
        with self._lock:
            self._model_labels = cleaned

    def set_model_loaded(self, loaded: bool) -> None:
        with self._lock:
            self._model_loaded = 1 if loaded else 0

    def inc_inflight(self) -> None:
        with self._lock:
            self._inflight += 1

    def dec_inflight(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = max(0, int(depth))

    def inc_stream_connection(self) -> None:
        with self._lock:
            self._stream_active += 1

    def dec_stream_connection(self) -> None:
        with self._lock:
            self._stream_active = max(0, self._stream_active - 1)

    @staticmethod
    def _esc(value: str) -> str:
        """Escape a Prometheus label value (backslash, quote, newline)."""
        return (
            str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        )

    def render(self) -> str:
        """Emit the Prometheus text exposition format."""
        with self._lock:
            lines: list[str] = []

            if self._model_labels:
                label_str = ",".join(
                    f'{k}="{self._esc(v)}"' for k, v in sorted(self._model_labels.items())
                )
                lines.append(
                    "# HELP adapter_model_info Model identity; the labels are the payload, the value is always 1."
                )
                lines.append("# TYPE adapter_model_info gauge")
                lines.append(f"adapter_model_info{{{label_str}}} 1")

            lines.append("# HELP adapter_infer_total Total inference calls by outcome and task.")
            lines.append("# TYPE adapter_infer_total counter")
            tasks_seen = sorted({t for (_o, t) in self._infer_total})
            for outcome in self.KNOWN_OUTCOMES:
                for task in tasks_seen:
                    count = self._infer_total.get((outcome, task))
                    if count is None and task != "":
                        continue          # don't fabricate zero series per task
                    lines.append(
                        f'adapter_infer_total{{outcome="{outcome}",task="{self._esc(task)}"}} {count or 0}'
                    )

            lines.append("# HELP adapter_infer_latency_seconds Inference latency histogram by task.")
            lines.append("# TYPE adapter_infer_latency_seconds histogram")
            for task in sorted(self._lat):
                hist = self._lat[task]
                tl = f'task="{self._esc(task)}"'
                for ub in self._latency_buckets:
                    lines.append(
                        f'adapter_infer_latency_seconds_bucket{{{tl},le="{ub}"}} {hist["buckets"][ub]}'
                    )
                lines.append(
                    f'adapter_infer_latency_seconds_bucket{{{tl},le="+Inf"}} {hist["inf"]}'
                )
                lines.append(f'adapter_infer_latency_seconds_sum{{{tl}}} {hist["sum"]}')
                lines.append(f'adapter_infer_latency_seconds_count{{{tl}}} {hist["count"]}')

            lines.append("# HELP adapter_model_loaded 1 if the model is loaded into memory.")
            lines.append("# TYPE adapter_model_loaded gauge")
            lines.append(f"adapter_model_loaded {self._model_loaded}")

            lines.append("# HELP adapter_stream_connections_active Active WebSocket streams.")
            lines.append("# TYPE adapter_stream_connections_active gauge")
            lines.append(f"adapter_stream_connections_active {self._stream_active}")

            lines.append("# HELP adapter_inflight_requests Requests currently being served.")
            lines.append("# TYPE adapter_inflight_requests gauge")
            lines.append(f"adapter_inflight_requests {self._inflight}")

            lines.append("# HELP adapter_queue_depth Requests waiting for the model.")
            lines.append("# TYPE adapter_queue_depth gauge")
            lines.append(f"adapter_queue_depth {self._queue_depth}")

            # Adapter-defined series, in registration order.
            for name, reg in self._custom_counters.items():
                lines.append(f"# HELP {name} {reg['help']}")
                lines.append(f"# TYPE {name} counter")
                if reg["label_key"] is None:
                    lines.append(f"{name} {reg['values'].get('', 0)}")
                else:
                    for lv in sorted(reg["values"]):
                        lines.append(
                            f'{name}{{{reg["label_key"]}="{self._esc(lv)}"}} {reg["values"][lv]}'
                        )
            for name, hist in self._custom_hists.items():
                lines.append(f"# HELP {name} {hist['help']}")
                lines.append(f"# TYPE {name} histogram")
                for ub in hist["ubs"]:
                    lines.append(f'{name}_bucket{{le="{ub}"}} {hist["buckets"][ub]}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {hist["inf"]}')
                lines.append(f"{name}_sum {hist['sum']}")
                lines.append(f"{name}_count {hist['count']}")

            return "\n".join(lines) + "\n"

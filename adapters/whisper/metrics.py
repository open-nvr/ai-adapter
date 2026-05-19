# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Prometheus exposition for the Whisper contract service.

Mirrors ``adapters/piper/metrics.py`` / ``adapters/yolov8/metrics.py``
with latency buckets tuned for ASR: a 30s clip on CPU can take 5-15s
on the base model, GPU pulls that under 1s. Buckets cover both regimes.

A2.3 will extract this into ``opennvr-adapter-sdk``. Symmetric changes
across all three adapters until then.
"""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    """In-process metrics registry. Thread-safe."""

    # Whisper inference latency tracks audio duration × model size.
    # base@CPU is ~1.5x realtime; large@GPU is ~0.05x realtime. Buckets
    # span 10ms (cached short clip) → 60s (long clip on CPU).
    LATENCY_BUCKETS_SECONDS: tuple[float, ...] = (
        0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
    )

    KNOWN_OUTCOMES: tuple[str, ...] = (
        "ok",
        "model_error",
        "provider_error",
        "transport_error",
        "refused",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._infer_total: dict[str, int] = defaultdict(int)
        self._latency_bucket_counts: dict[float, int] = {
            ub: 0 for ub in self.LATENCY_BUCKETS_SECONDS
        }
        self._latency_inf_count: int = 0
        self._latency_sum: float = 0.0
        self._latency_count: int = 0
        self._model_loaded: int = 0
        self._stream_active: int = 0
        self._inflight: int = 0
        self._queue_depth: int = 0

        for outcome in self.KNOWN_OUTCOMES:
            self._infer_total[outcome] = 0

    def record_infer(self, outcome: str, latency_seconds: float) -> None:
        if outcome not in self.KNOWN_OUTCOMES:
            outcome = "model_error"
        with self._lock:
            self._infer_total[outcome] += 1
            for ub in self.LATENCY_BUCKETS_SECONDS:
                if latency_seconds <= ub:
                    self._latency_bucket_counts[ub] += 1
            self._latency_inf_count += 1
            self._latency_sum += latency_seconds
            self._latency_count += 1

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

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []

            lines.append("# HELP adapter_infer_total Total inference calls by outcome.")
            lines.append("# TYPE adapter_infer_total counter")
            for outcome in self.KNOWN_OUTCOMES:
                lines.append(
                    f'adapter_infer_total{{outcome="{outcome}"}} {self._infer_total.get(outcome, 0)}'
                )

            lines.append("# HELP adapter_infer_latency_seconds Inference latency histogram.")
            lines.append("# TYPE adapter_infer_latency_seconds histogram")
            for ub in self.LATENCY_BUCKETS_SECONDS:
                lines.append(
                    f'adapter_infer_latency_seconds_bucket{{le="{ub}"}} {self._latency_bucket_counts[ub]}'
                )
            lines.append(
                f'adapter_infer_latency_seconds_bucket{{le="+Inf"}} {self._latency_inf_count}'
            )
            lines.append(f"adapter_infer_latency_seconds_sum {self._latency_sum}")
            lines.append(f"adapter_infer_latency_seconds_count {self._latency_count}")

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

            return "\n".join(lines) + "\n"

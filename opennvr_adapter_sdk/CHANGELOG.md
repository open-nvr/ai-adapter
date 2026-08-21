# Changelog

All notable changes to `opennvr-adapter-sdk` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the SDK uses semantic versioning aligned with the AI Adapter Contract major version (SDK v1.x targets contract v1).

## [1.2.0] — 2026-08

### Added

- **Model identity + per-task labels on `/metrics`** (observability slice 1).
  `adapter_model_info{adapter, adapter_version, model, model_version,
  framework, fingerprint} 1` exports the model's identity as an info-metric,
  populated from the service's existing `model_info()` at lifespan startup
  and refreshed on every `/capabilities` build — so §11.3 fingerprint drift
  is visible on `/metrics` too, and a latency regression can be correlated
  with a weights change from one scrape. `adapter_infer_total` and
  `adapter_infer_latency_seconds` gain a `task` label (task per request,
  from the payload). The task label set is **closed**: only
  `tasks_advertised` values become series; anything else folds into
  `"other"` (task strings are client-controlled — an open set would let any
  client mint unbounded series), and `task=""` covers unattributed calls
  (stream frames, transport errors). No adapter changes required — every
  SDK adapter gets all of this by rebuilding against 1.2.0.

### Changed

- `Metrics(known_tasks=...)` constructor parameter and
  `record_infer(..., task="")` keyword (both optional — existing callers
  are unaffected). Exposition format: the infer counter and latency
  histogram series now always carry the `task` label; consumers matching
  exact series strings should match on the metric-name prefix and sum
  across labels.

## [1.1.0] — 2026-07

### Added

- `opennvr_adapter_sdk.model_fetch.ensure_model_file(path, url, *, label,
  logger)` — first-boot model download for adapters whose weights ship
  outside the image (whisper-adapter pattern, now shared). Stdlib-only
  (urllib); streams to `<path>.part` and renames on success so a killed
  container never leaves truncated weights; a present file always wins so
  offline / sovereignty-strict installs that pre-populate the weights
  volume never trigger egress. Adopted by the llamacpp, whispercpp,
  pipertts, and smolvlm adapters.

## [1.0.0] — 2026-05

First public release. Extracted from the three reference adapters
(Piper TTS, YOLOv8 object detection, Whisper ASR) shipped in
`open-nvr/ai-adapter`. All §3 endpoints, §5 result conventions,
§6 WebSocket streaming protocol, §7 failure envelope, and §11.3
fingerprint drift detection are exercised by the three references.

### Added

- `AdapterService` ABC with four required abstract methods
  (`load`, `is_ready`, `fingerprint`, `model_info`,
  `hardware_evaluation`, `infer`) plus optional `handle_stream`
  for streaming adapters.
- `AdapterApp` builder that wraps an `AdapterService` in a
  FastAPI app implementing all six mandatory contract endpoints
  (`/health`, `/capabilities`, `/hardware/evaluation`,
  `/metrics`, `/infer`, `/infer/stream`), plus auth +
  correlation_id middleware, Prometheus metrics, lifespan
  startup, and body parsing for `BodyShape.{TEXT, IMAGE, AUDIO, GENERIC}`.
- `ServiceError` exception that translates to the §7 failure
  envelope.
- `BODY_BYTES_KEY` constant for the binary body payload key;
  caller-supplied params that shadow it are rejected with
  `malformed_input` rather than being silently overwritten.
- `opennvr_adapter_sdk.contract` submodule with every Pydantic
  wire type the contract defines; commonly-used types are
  re-exported at the package root.
- Oversize bodies (`max_body_bytes`) return HTTP 413, not 400.
- Streaming adapters automatically get `inc/dec_stream_connection`
  metrics; per-frame metrics via `self.metrics`.
- `service_factory` constructor parameter for lazy service
  construction at lifespan startup — useful for test fixtures
  that monkey-patch `__init__`.

### Notes

- Apache-2.0 licensed so third parties can write closed-source
  production adapters. The reference adapters and the ai-adapter
  app are AGPL-3.0; the SDK boundary stays permissive.
- Pinned to FastAPI ≥0.115 / Pydantic ≥2.7 / Python ≥3.10. Loosen
  the upper bounds in your fork if you need wider compatibility.

[1.0.0]: https://github.com/open-nvr/ai-adapter/releases/tag/sdk-v1.0.0

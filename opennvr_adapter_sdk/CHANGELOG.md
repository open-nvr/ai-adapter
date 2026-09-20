# Changelog

All notable changes to `opennvr-adapter-sdk` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the SDK uses semantic versioning aligned with the AI Adapter Contract major version (SDK v1.x targets contract v1).


## [Unreleased]

### Added

- **Conformance kit knows `package_detection`** — a sample multipart frame
  and a JSON params fallback for the task, so an adapter advertising it
  gets the real `infer` check instead of a "nothing to send" WARN. Same
  1×1 JPEG the other vision tasks use; the check is on the wire shape.

- **`Metrics.gauges()`** — a read-only snapshot of the live inflight and
  queue-depth gauges (plus active stream connections). `render()` already
  published these to Prometheus, but §6.4's `stats` control message has to
  answer a streaming client with the same two numbers over the WebSocket,
  and a stream handler should not have to keep a second, drifting copy of a
  counter the SDK is already maintaining. Purely additive.


## [1.3.0] — 2026-09-11

### Added

- **`Adapter` — the front door.** Declare the model, decorate the loader
  and the inference handler; the SDK derives the fingerprint (from the
  weights file, and never null), health, the hardware verdict, the
  modalities, the body shape and the error taxonomy. The scaffolded
  service was 218 lines of TODO before wrapping a single model. It
  compiles to an ordinary `AdapterService`, so existing adapters are
  untouched.
- **A real OpenAPI 3.1 document.** `AdapterApp` was always FastAPI, so
  `/openapi.json` always existed — and documented nothing: every route
  returned a bare `JSONResponse`, leaving six paths, zero schemas and
  zero components. Now every response is typed from the contract models,
  `/infer`'s request body is described for the adapter's own
  `BodyShape`, every error status carries the §7 failure envelope,
  `/metrics` is declared as Prometheus text, and bearer auth is declared
  where the middleware enforces it. 0 schemas → 22.
- **AsyncAPI 3.0 at `/asyncapi.json`** for `/infer/stream`, which
  OpenAPI cannot express — all ten §6 message types, generated from the
  contract models the session exchanges.
- **`opennvr-adapter`, a packaged CLI**: `new` scaffolds a STANDALONE
  adapter project (the old `scaffold.sh` wrote into the ai-adapter
  repository's own `adapters/` directory), `dev` drives the adapter
  in-process with no stack, `validate` runs the full conformance suite
  against a directory, `conform` against a running adapter, and `spec`
  emits either document.
- **The conformance kit ships in the wheel** as
  `opennvr_adapter_sdk.conformance`. It previously lived only in the
  ai-adapter repository, so the one tool that tells a model developer
  their adapter will be accepted was unreachable by anyone who had
  merely installed the SDK. The repo-root `conformance` package remains
  as a re-export.
- **A cookbook** — nine runnable files, one per class, imported and
  exercised by the test suite so an example that names something the
  SDK no longer exports breaks in CI.
- **A published reference** at `opennvr.org/adapters`
  (mkdocs-material + mkdocstrings, `make sdk-site`), and
  `API_TIERS` — the thirty exports now have a documented front door,
  with `__all__` assembled from the tiers so the two cannot drift.
- **`opennvr-adapter listing`** — generates the `adapters_index.yml`
  entry that makes an adapter installable, from its own
  `/capabilities`, so a listing cannot claim a task it does not
  advertise. See open-nvr's `docs/CONTRIBUTING_ADAPTERS.md`.

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

# Changelog

All notable changes to **OpenNVR AI Adapter** (the reference adapter server in
this repo) are documented here. The standalone Python SDK has its own
changelog under [`opennvr_adapter_sdk/CHANGELOG.md`](opennvr_adapter_sdk/CHANGELOG.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — targeting v0.1.0

First public release of the reference adapter server. Aligned with the
v0.1.0 cut of the [OpenNVR](https://github.com/open-nvr/open-nvr) NVR and the
`opennvr-adapter-sdk` v1.0.0 published to PyPI.

### Added

#### SDK & contract

- **`opennvr-adapter-sdk` published on PyPI.** Apache-2.0 SDK that adapter
  authors install to write a new contract-compliant detector in ~30 lines.
  Public API: `AdapterService` ABC, `AdapterApp` FastAPI builder,
  `ServiceError` envelope, `BodyShape` enum, and re-exports of every Pydantic
  wire type from the contract.
- **AI Adapter Contract v1 compliance.** All six mandatory endpoints
  (`/health`, `/capabilities`, `/hardware/evaluation`, `/metrics`, `/infer`,
  `/infer/stream`), the five result conventions, the WebSocket streaming
  protocol, the typed failure envelope, and live model fingerprint drift
  detection are exercised by the reference adapters.
- **Conformance test suite** in `conformance/` for verifying any
  contract-compliant adapter, in-tree or third-party.

#### Reference adapters

The reference server ships with the following adapters auto-discovered at
startup:

- **YOLOv8** — ONNX object detection, CPU-friendly, no torch dependency.
- **YOLOv11** — PyTorch person counting with ByteTrack.
- **InsightFace** — face detection and recognition (Buffalo-L ArcFace).
- **BLIP** — scene captioning.
- **HuggingFace** — cloud inference proxy.
- **Whisper** — speech-to-text via faster-whisper (CPU or GPU).
- **Piper** — text-to-speech via ONNX voices.
- **Ollama** — local LLM access over HTTP.

#### Architecture

- **Auto-discovery + lazy loading.** Place an adapter file under
  `app/adapters/`, restart, the system finds it automatically. Models load
  into memory only when their first request arrives. Idle adapters consume
  zero memory.
- **Anti-bloat dependencies.** A minimal core install is roughly nine packages
  (~50 MB). Adapter ML libraries live in `[project.optional-dependencies]`
  groups so users only install what their deployment uses. The full install
  (`uv sync --extra all --extra cpu`) is ~4 GB; most deployments need
  `--extra yolo --extra face` (~750 MB).
- **Graceful adapter skipping.** If an adapter's optional dependencies are
  missing, `PluginManager` skips it cleanly with a helpful install hint
  rather than crashing the server.
- **Pydantic-validated response envelopes** catch invalid data at the
  boundary — confidence out of range, missing fields, count mismatches.
- **Async pipeline engine.** Chain multiple tasks sequentially via
  `POST /pipeline/run`. Output of step N feeds into step N+1.
- **Process isolation by default.** Each adapter can run as its own container
  with its own resource limits. A crash in one adapter doesn't take down the
  alerts pipeline.

#### Operations

- **Multi-stage Docker build** with CPU and GPU variants
  (`docker build --build-arg USE_GPU=true`).
- Built-in `HEALTHCHECK` against `/health` every 30 seconds.
- **Opt-in `X-API-Key` authentication** via environment variables.

### Security

- **Sovereignty enforcement at the middleware boundary.** KAI-C (in the
  `open-nvr` repo) refuses to register adapters that declare
  `network_egress` permissions under the default `local_only` policy.
- **Operator-gated permissions.** GPU, host filesystem, network egress, and
  shared-memory access are all declared in `/capabilities` and surfaced for
  approval rather than implicit.
- **Live fingerprint drift detection** built into the SDK. The middleware
  polls `/capabilities` every 60 seconds and emits an audit event when the
  declared fingerprint changes.
- **Typed error envelope.** Every failure path raises `ServiceError`, which
  the SDK translates to a wire-compatible response with category, code,
  transient flag, and `retry_after_ms`.

### Changed

- Migrated FastAPI lifecycle from the deprecated `@app.on_event("startup")`
  decorator to the lifespan async context manager pattern.

### License

The reference server is licensed under [GNU AGPL v3.0](LICENSE). The SDK
(`opennvr_adapter_sdk/`) is licensed under **Apache-2.0** so adapter authors
can publish under any compatible license, including proprietary. Model
weights are not AGPL-bound — ship them under whatever licence the model
permits.

---

[Unreleased]: https://github.com/open-nvr/ai-adapter/compare/...HEAD

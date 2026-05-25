# Changelog

All notable changes to **OpenNVR AI Adapter** (the reference adapter server in
this repo) are documented here. The standalone Python SDK has its own
changelog under [`opennvr_adapter_sdk/CHANGELOG.md`](opennvr_adapter_sdk/CHANGELOG.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — targeting v0.1.0

First public release of the reference adapter server. Aligned with the
v0.1.0 cut of the [OpenNVR](https://github.com/open-nvr/open-nvr) NVR. The
`opennvr-adapter-sdk` is structured for PyPI distribution; the upload wires
off the first `sdk-v*` tag, so until then install from source.

### Added

#### SDK & contract

- **`opennvr-adapter-sdk`.** Apache-2.0 SDK that adapter authors install to
  write a new contract-compliant detector in ~30 lines. Public API:
  `AdapterService` ABC, `AdapterApp` FastAPI builder, `ServiceError`
  envelope, `BodyShape` enum, and re-exports of every Pydantic wire type
  from the contract. PyPI publish wires off the first `sdk-v*` tag; until
  then, install from source via `pip install -e ./opennvr_adapter_sdk`.
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
- **InsightFace** — face detection, recognition, and embedding extraction
  (Buffalo-L ArcFace). SDK-based; accepts multipart / base64 frames.
  Exposes face-DB CRUD (`/faces/register`, `/faces`, `/faces/{id}`) so
  external apps like the Smart Doorbell example can enroll known faces
  via REST without a shared volume. Embeddings live in a JSON-file
  face DB at `OPENNVR_INSIGHTFACE_FACE_DB`; raw face images are never
  persisted.
- **BLIP** — scene captioning.
- **HuggingFace** — cloud inference proxy.
- **Whisper** — speech-to-text via faster-whisper (CPU or GPU).
- **Piper** — text-to-speech via ONNX voices.
- **fast-plate-ocr** — license-plate recognition (Apache-2.0 upstream,
  ONNX, CPU-only, plate-specific). New `lpr` install extra; single-
  purpose OCR adapter designed to be chained downstream of YOLOv8 by
  the `license-plate-recognition` example app on OpenNVR. The adapter
  decodes request body bytes (JPEG / PNG / WebP / BMP via OpenCV)
  into a numpy array before invoking
  `LicensePlateRecognizer.run()`, since the upstream library's 1.x
  API accepts paths or ndarrays but not raw bytes. Garbage bytes
  surface as a typed `TRANSPORT_ERROR(invalid_image)` envelope with
  HTTP 400.
- **Ollama** — local LLM access over HTTP. Supports the OpenAI-style
  `tools` / `tool_choice` request fields and normalises Ollama's
  native tool-call response into the OpenAI-shaped `message.tool_calls`
  array (with synthesised call ids and JSON-stringified arguments)
  so Pipecat / OpenAI clients work without translation. Tool support
  requires a tool-capable model (Llama 3.1+, Qwen 2.5+, Mistral Nemo,
  others — see https://ollama.com/blog/tool-support).

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

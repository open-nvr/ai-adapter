# Changelog

All notable changes to **OpenNVR AI Adapter** (the reference adapter server in
this repo) are documented here. The standalone Python SDK has its own
changelog under [`opennvr_adapter_sdk/CHANGELOG.md`](opennvr_adapter_sdk/CHANGELOG.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`package-detection` adapter** — contract-v1 `package_detection` on
  port 9010: parcels on a doorstep, the one class COCO does not have.
  A YOLOv8n fine-tuned on two openly licensed doorstep sets (*package at
  front door*, 1,293 images, MIT; Roboflow's *Packages*, CC0) on
  `onnxruntime`, returning §5.1 `detections` with the label `package`,
  normalized boxes un-letterboxed into the source frame, plus `count`
  and a `labels` echo. Advertises ONLY `package_detection` — a
  one-class model must not satisfy an app whose `object_detection`
  means people and vehicles. On demand over HTTP `/infer`, no stream:
  its consumer (open-nvr's `package-delivery` app) asks once when
  Tier-0 says somebody left the doorstep and on a slow recheck cadence,
  and picks this adapter over VQA and over the COCO bag-class stand-in
  the moment it registers. Trained 40 epochs at 416 px on 1,242
  images; on the held-out test split precision 0.974, recall 0.804,
  mAP50 0.914, mAP50-95 0.631 — precise before it is complete, which
  is the right way round for a consumer that re-counts on a cadence
  and must not invent a delivery. CPU is the design target: 11.7 MB
  ONNX, 23–34 ms per frame on two x86 cores. Per-call `conf`, `iou`, `imgsz`,
  `max_detections`; multi-class fine-tunes work unchanged through
  `PACKAGE_DETECTION_LABELS`. Domain metrics
  `adapter_package_frames_total{result=packages|empty}` and
  `adapter_packages_total{label}`. Weights are not baked: mounted, or
  fetched once from `PACKAGE_DETECTION_MODEL_URL` (the release asset),
  with `network_egress` derived from that setting. The training and
  export recipe ships as `adapters/package_detection/train_package_model.py`,
  including the hook for fine-tuning on a site's own frames.

- **`yolo-pose` adapter** — contract-v1 `pose_estimation` on port 9009,
  wrapping a YOLO11n-pose ONNX model on `onnxruntime`. Returns a
  COCO-17 skeleton per person (`persons[].keypoints` — 17
  `[x, y, confidence]` triples in the fixed COCO order, plus
  `keypoint_names` and `frame_dimensions`), over HTTP `/infer` or the
  full §6 WebSocket streaming protocol, which is the path a 10 fps
  camera uses. Built for apps that reason about body geometry rather
  than object location: wand-compliance verification (is the guard's
  detector tracking the visitor's arms?), fall detection, queue
  orientation.
  CPU-first by design — the nano model at a 448 px default input, one
  serial session, ~47 ms/frame and ~160 MB RSS measured on an 8-core
  x86 CPU. Coordinates are **pixels in the source frame**, not §5.1's
  normalized boxes: normalizing a 16:9 frame to a unit square distorts
  exactly the limb angles these apps measure. Per-call `conf`, `iou`,
  `imgsz` and `max_persons`; per-joint visibility exported as a domain
  metric (`adapter_pose_keypoints_visible_total{keypoint=…}`), which
  is how a camera that has stopped seeing wrists shows up before the
  app goes quiet. Weights are not baked into the image: the ONNX is
  exported locally (`python download_models.py --all`) and mounted,
  or fetched once from an operator-configured `YOLO_POSE_MODEL_URL` —
  and `permissions.network_egress` is derived from that setting, so the
  default deployment declares no egress at all and a configured fetch
  declares exactly its one host.

## [0.1.5] — 2026-09-10

The LPR adapter goes from unusable to correct. Everything below is the
same plate-reading path, fixed in four steps — packaging, localization,
confidence and the box's frame of reference. `opennvr-adapter-sdk` is
unchanged at 1.2.0 (already on PyPI); it is versioned by its own API
contract, not by this repo's release number.

### Fixed

- **The shipped LPR image could never read a plate.** `fast-plate-ocr`
  was installed without its `[onnx]` extra, so the runtime the model
  needs was absent from the published image — every read failed on a
  machine that was otherwise correctly configured.
- **LPR confidence was ALWAYS 1.0**, which made every precision floor in
  the adapter inert. The output parser expected a `.confidence`
  attribute, but `fast-plate-ocr` 1.1.0 (the shipped version) reports
  `char_probs` and has no `.confidence` — so `getattr` defaulted every
  production read to certainty: garbage like `'1023'` shipped as
  accepted, and raising the floor changed nothing. Confidence is now the
  **minimum** per-character probability (a read is only as trustworthy as
  its shakiest character); measured on the real models, a clean crop
  scores 0.997, noise-scene garbage 0.071 and a tiny-crop hallucination
  0.384 — min separates them where mean would blur them together. An
  unrecognised confidence shape now parses as UNTRUSTED (0.0) rather than
  1.0, so a read whose confidence cannot be seen can never outrank the
  floor. The unit tests missed this for a release because the stub
  returned `(text, conf)` tuples that parse fine — classic
  stub-blindness; the regression test now uses the real 1.1.0 shape and
  is sabotage-verified against the old behaviour.

### Added

- **Plate LOCALIZATION before OCR** — the missing middle stage.
  `fast-plate-ocr` is a pure OCR model expecting a single plate crop, but
  the platform's enrichment path sends the whole-VEHICLE best-frame crop;
  pointed at an entire vehicle it reads whatever character-like texture it
  finds, so nearly everything landed under the floor (silent misses) and
  the occasional hallucination landed over it (false reads). The adapter
  now localizes the plate with `open-image-models` (same author as the OCR
  model, same ONNX/CPU footprint, ~7 MB weights cached the same way) and
  OCRs the crop plus a small context margin. Measured end to end on a
  synthetic scene: whole-image OCR read `'L330'` at ~15% per-character
  confidence; detect → crop → OCR read the exact plate at 99.9%.
  Additive and fail-soft — a plate that isn't found falls back to
  whole-image OCR at a raised floor (default 0.75), a detector that fails
  to load never takes the adapter down (OCR is the core, detection the
  enhancer), and `OPENNVR_LPR_DETECTOR=""` selects explicit pure-OCR mode
  for callers who assert they send crops. Results gain
  `plate_detection {attempted, found, confidence, box, model_id}`.
- **Tiny-plate guard.** A localized plate narrower than
  `OPENNVR_LPR_MIN_PLATE_PX` (default 40) returns a clean non-read
  (`accepted=false`, `plate_detection.too_small`) instead of OCRing pixel
  soup — a distant car should produce nothing, not a wrong plate that
  fires a false unknown-vehicle alarm downstream.
- **`plate_detection.image_size`** — the box's frame of reference travels
  with the box. Consumers rejecting clipped (partial) reads judge the
  plate box against the image it was measured in (open-nvr#378), and
  multi-frame OCR sends candidate crops whose size differs from the
  visit's evidence frame, so a consumer guessing the denominator measures
  in the wrong image. The adapter knows exactly what it OCR'd — the
  `[width, height]` of the caller's bytes — so it now says so, and no
  consumer has to parse a JPEG header or guess.

### Changed

- **LPR precision floors are env-tunable**, so operators dial precision
  against recall without a rebuild: `OPENNVR_LPR_MIN_CONFIDENCE`
  (default raised 0.30 → 0.45 under min-character scoring) and
  `OPENNVR_LPR_MIN_PLATE_PX` (default 40).

## [0.1.4] — 2026-08-24

### Added

- **`ollamavlm` adapter** — contract-v1 visual_qa / scene_caption served
  by proxying to an Ollama endpoint (``OPENNVR_OLLAMA_VLM_URL``, default
  ``http://host.docker.internal:11434``; model
  ``OPENNVR_OLLAMA_VLM_MODEL``, default ``moondream``). Drop-in for the
  camera-agent's ``CAPTION_ADAPTER`` slot. Motivation: on macOS/Windows
  the Docker VM has no GPU access, so in-container VQA is CPU-only —
  pointing this adapter at a host-side Ollama gets Metal/GPU inference
  while every call stays inside the audited Adapter Contract, and model
  management collapses to ``ollama pull``. Lazy-ready by design (an
  unreachable endpoint is a transient per-infer error, not a boot
  failure), auto-pulls a missing model via Ollama's own API, ~80 MB
  image with no weights and no ML deps.
- **Model identity + per-task + domain metrics on `/metrics`** (SDK
  1.2.0, wired into every published adapter).
  `adapter_model_info{adapter, adapter_version, model, model_version,
  framework, fingerprint} 1` exports the served model's identity, so
  §11.3 fingerprint drift is visible from one scrape and a latency
  regression can be correlated with a weights change.
  `adapter_infer_total` / `adapter_infer_latency_seconds` gain a
  closed-set `task` label, and adapters register model-specific domain
  metrics (detection counts, plate-read confidence, …) through
  `load()`-time registration.

### Removed

- **The camera-agent-lite adapters** (`llamacpp`, `smolvlm`,
  `whispercpp`, `pipertts`) — companion to open-nvr's camera-agent-lite
  removal. llamacpp/smolvlm were amd64-only (blocked upstream on
  ggml-org/llama.cpp#19177 — the ARM audience lite targeted could never
  run them natively), and whispercpp/pipertts duplicated the `whisper`
  and `piper` adapters. STT/TTS stay covered by whisper/piper; the
  llama.cpp runtime stays covered through Ollama (`ollamavlm` + the
  agent's host-Ollama path). Every remaining published adapter image is
  multi-arch.

## [0.1.3] — 2026-08-17

ARM64 release. Adapter images are now `linux/amd64` + `linux/arm64`
manifest lists, fixing the Apple Silicon / Raspberry Pi 5 install
failure where a fresh OpenNVR `docker compose pull` aborted with
``no matching manifest for linux/arm64/v8`` on the first adapter image
it reached. (No 0.1.2 was ever tagged; this release follows 0.1.1.)

### Added

- **linux/arm64 images for 11 of 13 adapters** — `yolov8`, `piper`,
  `pipertts`, `whisper`, `whispercpp`, `fast-plate-ocr`, `insightface`,
  `blip`, `bytetrack`, `moondream`, and `voice` now publish multi-arch
  manifest lists. Every pinned dependency was verified to resolve to a
  pre-built ``manylinux*_aarch64`` wheel (no source builds under QEMU).
  `llamacpp` and `smolvlm` remain amd64-only: they derive from
  ``ghcr.io/ggml-org/llama.cpp:server``, which publishes no arm64
  manifest upstream (ggml-org/llama.cpp#19177).
- ``platforms`` is now a per-adapter field on the publish matrix, so
  arch support is declared next to the adapter it belongs to.
- **SDK release-alignment guard** — a ``v*`` release run now fails if
  the SDK version pinned in the tree is not on PyPI (the gap v0.1.1
  shipped with: tree said 1.1.0, PyPI had only 1.0.0). Version numbers
  stay decoupled (SDK tracks the contract major); the guard aligns the
  *release*, not the number. See ``opennvr_adapter_sdk/RELEASING.md``.

### Fixed

- **blip / voice-bundle arm64 builds** — both pinned
  ``torch==2.9.1+cpu`` from ``download.pytorch.org/whl/cpu``, an index
  with no aarch64 ``+cpu`` wheel (pytorch/pytorch#136275). The
  Dockerfiles now branch on ``TARGETARCH``: arm64 installs PyPI's plain
  ``torch==2.9.1`` aarch64 wheel, which is already CPU-only (all CUDA
  deps carry ``platform_machine == "x86_64"`` markers). The amd64 path
  is unchanged.

### Documentation

- README: supported-architecture matrix and the maintainer rule that a
  dependency bump must keep aarch64 wheel coverage, since a regression
  only surfaces as an arm64 build timing out under QEMU.

## [0.1.1] — 2026-07-31

Aligned with the v0.1.1 cut of the OpenNVR NVR — this release supplies
the adapter set behind the new `camera-agent-lite` example.

### Added

- **Four new adapters for camera-agent-lite** — `llamacpp` (LLM via
  llama.cpp), `whispercpp` (speech-to-text), `pipertts` (text-to-speech),
  and `smolvlm` (vision). All four download their models on first boot
  via the SDK 1.1.0 `ensure_model_file` helper (a pre-populated weights
  volume always wins, so offline installs never trigger egress) and are
  in the GHCR publish matrix.
- **AI Adapter Contract v1.1 (optional detector spec)** — adapters can
  now declare their accelerator and expected input.
- SDK 1.1.0 — see
  [`opennvr_adapter_sdk/CHANGELOG.md`](opennvr_adapter_sdk/CHANGELOG.md).

### Changed

- CI publishes adapter images from every branch push
  (`:<branch>` + `:sha-<short>` tags), not just release tags.

### Fixed

- SDK `__version__` now matches `pyproject.toml` (1.1.0), unblocking the
  release version check.

### Documentation

- README opening section rewritten for search discoverability.

## [0.1.0] — 2026-07-14

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
- **BLIP** — scene captioning via HuggingFace
  ``Salesforce/blip-image-captioning-base`` (default; ``-large`` available
  by env override). SDK-based contract-compliant adapter — accepts JPEG /
  PNG / WebP / BMP request bodies and returns ``result.caption``. The
  camera-agent example's ``describe_camera`` tool calls this adapter via
  KAI-C. CPU-friendly (~2-4s/image on a modern 4-core); GPU-accelerated
  when ``torch.cuda.is_available()``. Operators with strict sovereignty
  pre-bake the model into the image or mount a populated HF cache.
- **HuggingFace** — cloud inference proxy.
- **Whisper** — speech-to-text via faster-whisper (CPU or GPU).
- **Piper** — text-to-speech via ONNX voices. Default response shape
  returns ``audio_uri`` (the internal ``opennvr://audio/...`` scheme,
  suitable for shared-volume deployments). Callers without a shared
  filesystem — e.g. the camera-agent example's Pipecat TTS service —
  send ``inline: true`` in the request body to also receive
  ``audio_b64`` inline alongside the URI.
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
- **ByteTrack** — multi-object tracking post-processor. Takes a frame's
  detections in (JSON, `BodyShape.TEXT`), returns the same detections
  with persistent `track_id` populated. Stateful per camera (TTL-evicted
  to keep memory bounded), tunable per-call via the `tracker_config`
  block. Composes with any detection-shaped upstream adapter — chain
  through KAI-C the same way the license-plate-recognition example
  chains YOLOv8 → fast-plate-ocr. Wraps
  [`supervision`](https://github.com/roboflow/supervision)'s
  ByteTrack (pinned `>=0.21,<0.30`; 0.30 removes ByteTrack from the
  package). The contract's `DetectionItem.track_id` field gains a
  populated home with this adapter — alert deduplication, dwell-time
  analytics, and track-stable state machines all become available
  to downstream examples.

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

[Unreleased]: https://github.com/open-nvr/ai-adapter/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/open-nvr/ai-adapter/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/open-nvr/ai-adapter/compare/v0.1.1...v0.1.3
[0.1.1]: https://github.com/open-nvr/ai-adapter/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/open-nvr/ai-adapter/releases/tag/v0.1.0

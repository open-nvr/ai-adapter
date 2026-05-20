# Changelog

All notable changes to `opennvr-adapter-sdk` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the SDK uses semantic versioning aligned with the AI Adapter Contract major version (SDK v1.x targets contract v1).

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
